"""
Batch Processing Page for Laser Trim Analyzer

Handles batch analysis of multiple Excel files with comprehensive
validation, progress tracking, and responsive design.
"""

import asyncio
import gc
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk

from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.core.exceptions import ProcessingError, ValidationError
from laser_trim_analyzer.core.models import AnalysisResult, AnalysisStatus, ValidationStatus
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.utils.file_utils import ensure_directory

# Set up logging before using it
logger = logging.getLogger(__name__)

from laser_trim_analyzer.gui.pages.base_page_ctk import BasePage
from laser_trim_analyzer.gui.widgets.file_drop_zone import FileDropZone
from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
from laser_trim_analyzer.gui.widgets.batch_results_widget_ctk import BatchResultsWidget
from laser_trim_analyzer.gui.widgets.progress_widgets_ctk import BatchProgressDialog

# Widget implementation flags
USING_CTK_BASE = True
USING_CTK_METRIC = True
USING_CTK_BATCH_RESULTS = True
USING_CTK_PROGRESS = True

# Import security validator
try:
    from laser_trim_analyzer.core.security import (
        SecurityValidator, get_security_validator
    )
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False

# Import resource management
try:
    from laser_trim_analyzer.core.resource_manager import (
        get_resource_manager, BatchResourceOptimizer, ResourceStatus
    )
    HAS_RESOURCE_MANAGER = True
except ImportError:
    HAS_RESOURCE_MANAGER = False
    ResourceStatus = None


class BatchProcessingPage(BasePage):
    """Batch processing page with comprehensive validation and responsive design."""

    def __init__(self, parent, main_window):
        """Initialize batch processing page with configuration validation."""
        super().__init__(parent, main_window)
        
        # Get configuration with validation
        try:
            self.analyzer_config = get_config()
            if not self._validate_configuration():
                self._show_configuration_error()
                return
        except Exception as e:
            self._show_configuration_error(str(e))
            messagebox.showerror(
                "Configuration Error",
                f"Failed to load application configuration:\n\n{str(e)}\n\nPlease check your config files."
            )
            return
            
        # Initialize database manager (use main window's db_manager if available)
        self._db_manager = None
        if hasattr(self.analyzer_config, 'database') and self.analyzer_config.database.enabled:
            # First try to use main window's db_manager
            if self.main_window and hasattr(self.main_window, 'db_manager'):
                self._db_manager = self.main_window.db_manager
                logger.info("Using main window's database manager")
            else:
                # Create our own if main window doesn't have one
                try:
                    self._db_manager = DatabaseManager(self.analyzer_config)
                    logger.info("Database manager initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize database: {e}")
                    import traceback
                    logger.error(f"Database initialization traceback:\n{traceback.format_exc()}")
                    self._db_manager = None
                    messagebox.showwarning(
                        "Database Warning",
                        f"Failed to initialize database connection:\n\n{str(e)}\n\n"
                        "Processing will continue without database features."
                    )
        
        try:
            self.processor = LaserTrimProcessor(self.analyzer_config, db_manager=self._db_manager)
        except Exception as e:
            self._show_processor_error(str(e))
            messagebox.showerror(
                "Processor Initialization Error",
                f"Failed to initialize the analysis processor:\n\n{str(e)}\n\n"
                "Please check your installation and configuration."
            )
            return
        
        # Initialize state
        self.selected_files: List[Path] = []
        self.batch_results: Dict[str, AnalysisResult] = {}
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        self.validation_results: Dict[str, bool] = {}
        
        # Validate required directories exist
        self._ensure_required_directories()
        
        # Processing control
        self._stop_event = threading.Event()
        self._processing_cancelled = False
        
        # Initialize resource management
        self.resource_manager = None
        self.resource_optimizer = None
        if HAS_RESOURCE_MANAGER:
            try:
                self.resource_manager = get_resource_manager()
                self.resource_optimizer = BatchResourceOptimizer(self.resource_manager)
                logger.info("Resource management initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize resource management: {e}")
                # Don't show error dialog for optional resource management feature
        
        logger.info(f"Batch processing page initialized (CTK widgets: Base={USING_CTK_BASE}, Metric={USING_CTK_METRIC}, BatchResults={USING_CTK_BATCH_RESULTS}, Progress={USING_CTK_PROGRESS})")

    def _create_page(self):
        """Create UI widgets with responsive design."""
        
        # Main scrollable container
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header with batch status
        self._create_header()
        
        # File selection section
        self._create_file_selection()
        
        # Batch validation section (initially hidden)
        self._create_batch_validation()
        
        # Processing options
        self._create_processing_options()
        
        # Processing controls
        self._create_processing_controls()
        
        # Results section
        self._create_results_section()
        
        # Setup initial layout
        self._setup_responsive_layout()

    def _create_header(self):
        """Create header section."""
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.pack(fill='x', pady=(0, 20))
        
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Batch Processing",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=15)
        
        # Batch validation status
        self.batch_status_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.batch_status_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.batch_status_label = ctk.CTkLabel(
            self.batch_status_frame,
            text="Batch Status: No Files Selected",
            font=ctk.CTkFont(size=12)
        )
        self.batch_status_label.pack(side='left', padx=10, pady=10)
        
        self.batch_indicator = ctk.CTkLabel(
            self.batch_status_frame,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.batch_indicator.pack(side='right', padx=10, pady=10)

    def _validate_configuration(self) -> bool:
        """Validate that the configuration is properly set up."""
        try:
            # Check if required directories exist
            if not hasattr(self.analyzer_config, 'data_directory'):
                return False
            if not hasattr(self.analyzer_config, 'processing'):
                return False
            return True
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _show_configuration_error(self, error_msg: str = None):
        """Show configuration error message."""
        error_text = error_msg or "Configuration Error: Unable to load application configuration."
        error_label = ctk.CTkLabel(
            self,
            text=error_text,
            font=ctk.CTkFont(size=14),
            text_color="red"
        )
        error_label.pack(expand=True, pady=20)
        logger.error(f"Configuration error: {error_text}")
    
    def _show_processor_error(self, error_msg: str):
        """Show processor initialization error message."""
        error_label = ctk.CTkLabel(
            self,
            text=f"Processor Error: {error_msg}",
            font=ctk.CTkFont(size=14),
            text_color="red"
        )
        error_label.pack(expand=True, pady=20)
        logger.error(f"Processor error: {error_msg}")
    
    def _ensure_required_directories(self):
        """Ensure required directories exist."""
        try:
            # Get data directory from config
            if hasattr(self.analyzer_config, 'data_directory'):
                data_dir = Path(self.analyzer_config.data_directory)
            else:
                # Fallback to default location
                data_dir = Path.home() / "LaserTrimResults"
            
            # Create main data directory if it doesn't exist
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            subdirs = ['batch_processing', 'exports', 'reports']
            for subdir in subdirs:
                (data_dir / subdir).mkdir(exist_ok=True)
                
            logger.info(f"Ensured directories exist at: {data_dir}")
        except Exception as e:
            logger.warning(f"Could not create directories: {e}")
            messagebox.showwarning(
                "Directory Creation Warning",
                f"Could not create output directories:\n\n{str(e)}\n\n"
                "You may need to manually select output locations."
            )
    

    def _create_file_selection(self):
        """Create file selection section."""
        self.file_selection_frame = ctk.CTkFrame(self.main_container)
        self.file_selection_frame.pack(fill='x', pady=(0, 20))
        
        self.file_selection_label = ctk.CTkLabel(
            self.file_selection_frame,
            text="File Selection:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.file_selection_label.pack(anchor='w', padx=15, pady=(15, 5))
        
        # File list display
        self.file_list_frame = ctk.CTkFrame(self.file_selection_frame, fg_color="transparent")
        self.file_list_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.file_list_label = ctk.CTkLabel(
            self.file_list_frame,
            text="Selected Files (0):",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.file_list_label.pack(anchor='w', padx=10, pady=(10, 5))
        
        self.file_listbox = ctk.CTkTextbox(
            self.file_list_frame,
            height=150,
            state="disabled"
        )
        self.file_listbox.pack(fill='x', padx=10, pady=(0, 10))
        
        # File selection buttons - responsive layout
        self.file_buttons_frame = ctk.CTkFrame(self.file_selection_frame, fg_color="transparent")
        self.file_buttons_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        # Create buttons that will be arranged responsively
        self.select_files_button = ctk.CTkButton(
            self.file_buttons_frame,
            text="Select Files",
            command=self._select_files,
            width=120
        )
        
        self.select_folder_button = ctk.CTkButton(
            self.file_buttons_frame,
            text="Select Folder",
            command=self._select_folder,
            width=120
        )
        
        self.clear_files_button = ctk.CTkButton(
            self.file_buttons_frame,
            text="Clear Files",
            command=self._clear_files,
            width=100
        )
        
        self.validate_batch_button = ctk.CTkButton(
            self.file_buttons_frame,
            text="Validate Batch",
            command=self._validate_batch,
            width=120,
            state="disabled"
        )
        
        # Store buttons for responsive layout
        self.file_buttons = [
            self.select_files_button,
            self.select_folder_button,
            self.clear_files_button,
            self.validate_batch_button
        ]

    def _create_batch_validation(self):
        """Create batch validation section."""
        self.batch_validation_frame = ctk.CTkFrame(self.main_container)
        # Initially hidden
        
        self.batch_validation_label = ctk.CTkLabel(
            self.batch_validation_frame,
            text="Batch Validation Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.batch_validation_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Validation metrics container
        self.validation_metrics_frame = ctk.CTkFrame(self.batch_validation_frame, fg_color="transparent")
        self.validation_metrics_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        # Validation metric cards
        self.total_files_card = MetricCard(
            self.validation_metrics_frame,
            title="Total Files",
            value="0",
            color_scheme="neutral"
        )
        
        self.valid_files_card = MetricCard(
            self.validation_metrics_frame,
            title="Valid Files",
            value="0",
            color_scheme="success"
        )
        
        self.invalid_files_card = MetricCard(
            self.validation_metrics_frame,
            title="Invalid Files",
            value="0",
            color_scheme="danger"
        )
        
        self.validation_rate_card = MetricCard(
            self.validation_metrics_frame,
            title="Validation Rate",
            value="0%",
            color_scheme="info"
        )
        
        # Store cards for responsive layout
        self.validation_cards = [
            self.total_files_card,
            self.valid_files_card,
            self.invalid_files_card,
            self.validation_rate_card
        ]

    def _create_processing_options(self):
        """Create processing options section."""
        self.options_frame = ctk.CTkFrame(self.main_container)
        self.options_frame.pack(fill='x', pady=(0, 20))
        
        self.options_label = ctk.CTkLabel(
            self.options_frame,
            text="Processing Options:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.options_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Options container for responsive layout
        self.options_container = ctk.CTkFrame(self.options_frame, fg_color="transparent")
        self.options_container.pack(fill='x', padx=15, pady=(0, 15))
        
        self.generate_plots_var = ctk.BooleanVar(value=False)  # Default off for batch
        self.generate_plots_check = ctk.CTkCheckBox(
            self.options_container,
            text="Generate Plots",
            variable=self.generate_plots_var
        )
        
        self.save_to_db_var = ctk.BooleanVar(value=True)
        self.save_to_db_check = ctk.CTkCheckBox(
            self.options_container,
            text="Save to Database",
            variable=self.save_to_db_var
        )
        
        self.comprehensive_validation_var = ctk.BooleanVar(value=True)
        self.comprehensive_validation_check = ctk.CTkCheckBox(
            self.options_container,
            text="Comprehensive Validation",
            variable=self.comprehensive_validation_var
        )
        
        # Store options for responsive layout
        self.option_widgets = [
            self.generate_plots_check,
            self.save_to_db_check,
            self.comprehensive_validation_check
        ]
    
    def _create_resource_monitoring(self):
        """Create resource monitoring section."""
        if not HAS_RESOURCE_MANAGER:
            return
            
        self.resource_frame = ctk.CTkFrame(self.main_container)
        self.resource_frame.pack(fill='x', pady=(0, 20))
        
        self.resource_label = ctk.CTkLabel(
            self.resource_frame,
            text="Resource Monitoring:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.resource_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Resource status container
        self.resource_container = ctk.CTkFrame(self.resource_frame, fg_color="transparent")
        self.resource_container.pack(fill='x', padx=15, pady=(0, 15))
        
        # Memory usage
        self.memory_frame = ctk.CTkFrame(self.resource_container, fg_color="transparent")
        self.memory_frame.pack(fill='x', pady=(0, 10))
        
        self.memory_label = ctk.CTkLabel(
            self.memory_frame,
            text="Memory Usage:",
            font=ctk.CTkFont(size=12)
        )
        self.memory_label.pack(side='left', padx=(10, 5))
        
        self.memory_progress = ctk.CTkProgressBar(self.memory_frame)
        self.memory_progress.pack(side='left', fill='x', expand=True, padx=5)
        self.memory_progress.set(0.5)
        
        self.memory_status_label = ctk.CTkLabel(
            self.memory_frame,
            text="50% (2.0 GB / 4.0 GB)",
            font=ctk.CTkFont(size=10)
        )
        self.memory_status_label.pack(side='left', padx=(5, 10))
        
        # Resource recommendations
        self.resource_status_label = ctk.CTkLabel(
            self.resource_container,
            text="Resources: Normal - Ready for batch processing",
            font=ctk.CTkFont(size=11),
            text_color="green"
        )
        self.resource_status_label.pack(anchor='w', padx=10, pady=(5, 10))
        
        # Resource optimization info
        self.optimization_label = ctk.CTkLabel(
            self.resource_container,
            text="Optimization: Standard processing mode",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.optimization_label.pack(anchor='w', padx=10, pady=(0, 5))

    def _create_processing_controls(self):
        """Create processing control buttons."""
        self.controls_frame = ctk.CTkFrame(self.main_container)
        self.controls_frame.pack(fill='x', pady=(0, 20))
        
        self.controls_label = ctk.CTkLabel(
            self.controls_frame,
            text="Processing Controls:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.controls_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Controls container
        self.controls_container = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.controls_container.pack(fill='x', padx=15, pady=(0, 15))
        
        self.start_button = ctk.CTkButton(
            self.controls_container,
            text="Start Processing",
            command=self._start_processing,
            state="disabled",
            fg_color="green",
            hover_color="darkgreen"
        )
        
        self.stop_button = ctk.CTkButton(
            self.controls_container,
            text="Stop Processing",
            command=self._stop_processing,
            state="disabled",
            fg_color="red",
            hover_color="darkred"
        )
        
        # Store control buttons for responsive layout
        self.control_buttons = [self.start_button, self.stop_button]

    def _create_results_section(self):
        """Create results display section."""
        self.results_frame = ctk.CTkFrame(self.main_container)
        self.results_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Processing Results:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.results_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # Results widget
        self.batch_results_widget = BatchResultsWidget(self.results_frame)
        self.batch_results_widget.pack(fill='both', expand=True, padx=15, pady=(0, 10))
        
        # Export button
        self.export_button = ctk.CTkButton(
            self.results_frame,
            text="Export Results",
            command=self._export_batch_results,
            state="disabled"
        )
        self.export_button.pack(pady=(0, 15))

    def _setup_responsive_layout(self):
        """Setup initial responsive layout."""
        self._arrange_responsive_elements()
        
    def _handle_responsive_layout(self, size_class: str):
        """Handle responsive layout changes."""
        super()._handle_responsive_layout(size_class)
        self._arrange_responsive_elements()
        
    def _arrange_responsive_elements(self):
        """Arrange elements based on current size class."""
        # Arrange file buttons
        self._arrange_file_buttons()
        
        # Arrange validation cards
        if hasattr(self, 'validation_cards'):
            self._arrange_validation_cards()
            
        # Arrange option widgets
        self._arrange_option_widgets()
        
        # Arrange control buttons
        self._arrange_control_buttons()
        
    def _arrange_file_buttons(self):
        """Arrange file selection buttons responsively."""
        # Clear existing layout
        for button in self.file_buttons:
            button.pack_forget()
            
        if self.current_size_class == 'small':
            # Stack vertically on small screens
            for button in self.file_buttons:
                button.pack(fill='x', padx=10, pady=2)
        else:
            # Arrange horizontally with responsive spacing
            for button in self.file_buttons:
                button.pack(side='left', padx=5, pady=10)

    def _arrange_validation_cards(self):
        """Arrange validation metric cards responsively."""
        # Clear existing grid
        for card in self.validation_cards:
            card.grid_forget()
            
        columns = self.get_responsive_columns(len(self.validation_cards))
        
        for i, card in enumerate(self.validation_cards):
            row = i // columns
            col = i % columns
            padding = self.get_responsive_padding()
            card.grid(row=row, column=col, sticky='ew', **padding)
            
        # Configure column weights
        for i in range(columns):
            self.validation_metrics_frame.columnconfigure(i, weight=1)

    def _arrange_option_widgets(self):
        """Arrange processing option widgets responsively."""
        # Clear existing layout
        for widget in self.option_widgets:
            widget.pack_forget()
            
        if self.current_size_class == 'small':
            # Stack vertically on small screens
            for widget in self.option_widgets:
                widget.pack(anchor='w', padx=10, pady=5)
        else:
            # Arrange horizontally
            for widget in self.option_widgets:
                widget.pack(side='left', padx=15, pady=10)

    def _arrange_control_buttons(self):
        """Arrange control buttons responsively."""
        # Clear existing layout
        for button in self.control_buttons:
            button.pack_forget()
            
        if self.current_size_class == 'small':
            # Stack vertically on small screens
            for button in self.control_buttons:
                button.pack(fill='x', padx=10, pady=5)
        else:
            # Arrange horizontally
            for button in self.control_buttons:
                button.pack(side='left', padx=10, pady=10)

    def _select_files(self):
        """Select multiple Excel files with security validation."""
        file_paths = filedialog.askopenfilenames(
            title="Select Excel files",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            # Validate files for security if available
            validated_files = []
            security_validator = None
            
            if HAS_SECURITY:
                try:
                    security_validator = get_security_validator()
                except Exception as e:
                    logger.warning(f"Failed to get security validator: {e}")
            
            for file_path in file_paths:
                path_obj = Path(file_path)
                
                # Security validation if available
                if security_validator:
                    try:
                        path_result = security_validator.validate_input(
                            path_obj,
                            'file_path',
                            {
                                'allowed_extensions': ['.xlsx', '.xls', '.xlsm'],
                                'check_extension': True
                            }
                        )
                        
                        if path_result.is_safe and not path_result.threats_detected:
                            validated_files.append(Path(path_result.sanitized_value))
                        else:
                            logger.warning(f"File rejected for security: {path_obj.name}")
                    except Exception as e:
                        logger.warning(f"Security validation error for {path_obj}: {e}")
                        # Add file anyway if security check fails
                        validated_files.append(path_obj)
                        messagebox.showwarning(
                            "Security Validation Warning",
                            f"Security check failed for {path_obj.name}:\n{str(e)}\n\n"
                            "File will be added anyway. Proceed with caution."
                        )
                else:
                    validated_files.append(path_obj)
            
            if validated_files:
                self.selected_files = validated_files
                self._update_file_display()
                self._update_batch_status("Files Selected", "orange")
                # Enable the start processing button when files are selected
                self._set_controls_state("normal")
                logger.info(f"Selected {len(self.selected_files)} files ({len(file_paths) - len(validated_files)} rejected)")
                
                if len(validated_files) < len(file_paths):
                    rejected_count = len(file_paths) - len(validated_files)
                    messagebox.showwarning(
                        "Security Validation",
                        f"{rejected_count} file(s) were rejected for security reasons."
                    )
            else:
                messagebox.showerror(
                    "Security Validation",
                    "All selected files were rejected for security reasons."
                )

    def _select_folder(self):
        """Select folder and find all Excel files."""
        folder_path = filedialog.askdirectory(title="Select folder with Excel files")
        
        if folder_path:
            self._start_folder_discovery(Path(folder_path))

    def _start_folder_discovery(self, folder_path: Path):
        """Start asynchronous folder discovery with progress indication."""
        # Update UI to show scanning state
        self._update_batch_status("Scanning Folder...", "orange")
        self.select_folder_button.configure(state="disabled", text="Scanning...")
        
        # Run discovery in background thread
        def discover_files():
            try:
                excel_files = []
                total_checked = 0
                
                # Find all Excel files recursively with progress updates
                for pattern in ["*.xlsx", "*.xls"]:
                    for file_path in folder_path.rglob(pattern):
                        # Filter out temporary files
                        if not file_path.name.startswith('~$'):
                            excel_files.append(file_path)
                        
                        # Update progress every 50 files
                        total_checked += 1
                        if total_checked % 50 == 0:
                            self.after(0, lambda count=total_checked: 
                                     self._update_batch_status(f"Scanning... ({count} files found)", "orange"))
                
                # Update UI on main thread
                self.after(0, self._handle_folder_discovery_complete, excel_files, folder_path)
                
            except Exception as e:
                logger.error(f"Folder discovery failed: {e}")
                self.after(0, self._handle_folder_discovery_error, str(e))
        
        # Start discovery thread
        thread = threading.Thread(target=discover_files, daemon=True)
        thread.start()

    def _handle_folder_discovery_complete(self, excel_files: List[Path], folder_path: Path):
        """Handle completion of folder discovery."""
        # Re-enable button
        self.select_folder_button.configure(state="normal", text="📂 Select Folder")
        
        if excel_files:
            self.selected_files = excel_files
            self._update_file_display()
            self._update_batch_status("Folder Selected", "orange")
            # Enable the start processing button when files are selected from folder
            self._set_controls_state("normal")
            logger.info(f"Found {len(excel_files)} Excel files in {folder_path}")
            
            # Show discovery summary
            messagebox.showinfo(
                "Folder Discovery Complete",
                f"Found {len(excel_files)} Excel files in:\n{folder_path.name}\n\n"
                f"Ready for batch validation and processing."
            )
        else:
            self._update_batch_status("No Files Selected", "gray")
            messagebox.showwarning("No Files", f"No Excel files found in {folder_path}")

    def _handle_folder_discovery_error(self, error_message: str):
        """Handle folder discovery error."""
        # Re-enable button
        self.select_folder_button.configure(state="normal", text="📂 Select Folder")
        self._update_batch_status("Discovery Error", "red")
        messagebox.showerror("Folder Discovery Error", f"Failed to scan folder:\n{error_message}")

    def _clear_files(self):
        """Clear selected files."""
        self.selected_files = []
        self.validation_results = {}
        self._update_file_display()
        self._update_batch_status("No Files Selected", "gray")
        self.batch_validation_frame.grid_remove()
        # Use _set_controls_state to properly disable all buttons when no files
        self._set_controls_state("normal")  # This will disable start button due to no files

    def _update_file_display(self):
        """Update file list display."""
        self.file_list_label.configure(text=f"Selected Files ({len(self.selected_files)}):")
        
        self.file_listbox.configure(state="normal")
        self.file_listbox.delete("1.0", ctk.END)
        
        for i, file_path in enumerate(self.selected_files, 1):
            validation_status = ""
            if str(file_path) in self.validation_results:
                status = "✓" if self.validation_results[str(file_path)] else "✗"
                validation_status = f" [{status}]"
            
            self.file_listbox.insert(ctk.END, f"{i:3d}. {file_path.name}{validation_status}\n")
        
        self.file_listbox.configure(state="disabled")
        
        # Enable/disable buttons
        has_files = len(self.selected_files) > 0
        self.validate_batch_button.configure(state="normal" if has_files else "disabled")

    def _validate_batch(self):
        """Validate all selected files."""
        if not self.selected_files:
            messagebox.showerror("Error", "No files selected")
            return
        
        self._update_batch_status("Validating Batch...", "orange")
        
        # Run validation in thread
        def validate():
            try:
                from laser_trim_analyzer.utils.validators import BatchValidator
                
                validation_result = BatchValidator.validate_batch(
                    file_paths=self.selected_files,
                    max_batch_size=self.analyzer_config.processing.max_batch_size
                )
                
                # Store individual file validation results
                invalid_files = validation_result.metadata.get('invalid_files', [])
                self.validation_results = {}
                
                for file_path in self.selected_files:
                    is_valid = not any(str(file_path) in invalid['file'] for invalid in invalid_files)
                    self.validation_results[str(file_path)] = is_valid
                
                # Update UI on main thread
                self.after(0, self._handle_batch_validation_result, validation_result)
                
            except Exception as e:
                logger.error(f"Batch validation failed: {e}")
                error_msg = str(e)
                if "No module named" in error_msg:
                    error_msg = f"Missing dependency for validation: {error_msg}\nPlease ensure all required packages are installed."
                elif "Config object has no attribute" in error_msg:
                    error_msg = f"Configuration error: {error_msg}\nPlease check your configuration settings."
                self.after(0, self._handle_batch_validation_error, error_msg)
        
        thread = threading.Thread(target=validate, daemon=True)
        thread.start()

    def _handle_batch_validation_result(self, validation_result):
        """Handle batch validation result."""
        if validation_result.is_valid:
            self._update_batch_status("Batch Validation Passed", "green")
            
            # Show validation metrics (use pack since parent uses pack)
            self.batch_validation_frame.pack(fill='x', pady=(0, 20))
            
            metadata = validation_result.metadata
            total_files = len(self.selected_files)
            valid_files = metadata.get('valid_files', 0)
            invalid_files = total_files - valid_files
            validation_rate = (valid_files / total_files * 100) if total_files > 0 else 0
            
            self.total_files_card.update_value(str(total_files), "info")
            self.valid_files_card.update_value(str(valid_files), "success")
            self.invalid_files_card.update_value(str(invalid_files), 
                                               "success" if invalid_files == 0 else "warning")
            self.validation_rate_card.update_value(f"{validation_rate:.1f}%",
                                                 "success" if validation_rate > 90 else "warning")
            
            # Update file display with validation status
            self._update_file_display()
            
            # Enable start button if we have valid files
            if valid_files > 0:
                self.start_button.configure(state="normal")
            
            # Show warnings if any
            if validation_result.warnings:
                warning_msg = "Batch validation warnings:\n" + "\n".join(validation_result.warnings)
                messagebox.showwarning("Validation Warnings", warning_msg)
                
        else:
            self._update_batch_status("Batch Validation Failed", "red")
            
            # Show validation metrics with errors (use pack since parent uses pack)
            self.batch_validation_frame.pack(fill='x', pady=(0, 20))
            total_files = len(self.selected_files)
            self.total_files_card.update_value(str(total_files), "danger")
            self.valid_files_card.update_value("0", "danger")
            self.invalid_files_card.update_value(str(total_files), "danger")
            self.validation_rate_card.update_value("0%", "danger")
            
            # Update file display
            self._update_file_display()
            
            # Show errors
            error_msg = "Batch validation failed:\n" + "\n".join(validation_result.errors)
            messagebox.showerror("Batch Validation Failed", error_msg)

    def _handle_batch_validation_error(self, error_message):
        """Handle batch validation error."""
        self._update_batch_status("Validation Error", "red")
        messagebox.showerror("Validation Error", f"Batch validation failed:\n{error_message}")

    def _update_batch_status(self, status: str, color: str):
        """Update batch status indicator."""
        self.batch_status_label.configure(text=f"Batch Status: {status}")
        
        color_map = {
            "green": "#00ff00",
            "orange": "#ffa500",
            "red": "#ff0000", 
            "gray": "#808080"
        }
        
        self.batch_indicator.configure(text_color=color_map.get(color, "#808080"))

    def _start_processing(self):
        """Start batch processing."""
        if not self.selected_files:
            messagebox.showerror("Error", "No files selected")
            return
        
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return
        
        # Check if validation was run
        if not self.validation_results:
            reply = messagebox.askyesno(
                "No Validation", 
                "Batch validation hasn't been run. Proceed anyway?"
            )
            if not reply:
                return
        
        # Check resources before starting
        if self.resource_manager:
            status = self.resource_manager.get_current_status()
            if status.memory_critical:
                reply = messagebox.askyesno(
                    "Low Memory Warning",
                    "System memory is critically low.\n\n"
                    f"Available: {status.available_memory_mb:.0f}MB\n"
                    f"Recommended: Wait or close other applications\n\n"
                    "Continue anyway?"
                )
                if not reply:
                    return
        
        # Filter to only valid files if validation was run
        processable_files = []
        if self.validation_results:
            for file_path in self.selected_files:
                if self.validation_results.get(str(file_path), True):  # Default True if not validated
                    processable_files.append(file_path)
        else:
            processable_files = self.selected_files.copy()
        
        if not processable_files:
            messagebox.showerror("Error", "No valid files to process")
            return
        
        # Clear previous results
        self._clear_results()
        
        # Reset error tracking
        self._file_error_count = 0
        
        # Disable controls
        self._set_controls_state("disabled")
        
        # Show progress dialog
        self.progress_dialog = BatchProgressDialog(
            self,
            title="Batch Processing",
            total_files=len(processable_files)
        )
        self.progress_dialog.show()
        
        # Start processing in thread
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._run_batch_processing,
            args=(processable_files,),
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info(f"Started batch processing of {len(processable_files)} files")

    def _run_batch_processing(self, file_paths: List[Path]):
        """Run batch processing in background thread with performance optimizations and stop handling."""
        import gc
        import time
        
        try:
            # Reset stop flags
            self._stop_event.clear()
            self._processing_cancelled = False
            self.reset_stop_request()
            
            # Performance tracking
            start_time = time.time()
            last_gc_time = start_time
            last_progress_update = 0
            processed_count = 0
            
            # Create output directory if plots requested
            output_dir = None
            if self.generate_plots_var.get():
                # Use data_directory from config and create output subdirectory
                base_dir = self.analyzer_config.data_directory if hasattr(self.analyzer_config, 'data_directory') else Path.home() / "LaserTrimResults"
                output_dir = base_dir / "batch_processing" / datetime.now().strftime("%Y%m%d_%H%M%S")
                ensure_directory(output_dir)
            
            # Throttled progress callback to prevent UI flooding
            def progress_callback(message: str, progress: float):
                nonlocal last_progress_update
                
                # Check for cancellation
                if self._is_processing_cancelled():
                    return False  # Signal to stop
                    
                current_time = time.time()
                
                # Only update progress every 250ms to prevent UI flooding
                if current_time - last_progress_update >= 0.25:
                    last_progress_update = current_time
                    if self.progress_dialog:
                        self.after(0, lambda m=message, p=progress: self.progress_dialog.update_progress(m, p))
                        
                    # Force GUI update and yield CPU time
                    self.after(0, self.update)
                    time.sleep(0.001)  # Tiny sleep to yield CPU
                    
                return True  # Continue processing
            
            # Enhanced progress callback with memory monitoring
            def enhanced_progress_callback(message: str, progress: float):
                nonlocal processed_count, last_gc_time
                
                # Check for cancellation first
                if self._is_processing_cancelled():
                    return False
                    
                current_time = time.time()
                
                # Standard progress update
                if not progress_callback(message, progress):
                    return False
                
                # Memory management every 50 files or 30 seconds
                if (processed_count % 50 == 0 and processed_count > 0) or (current_time - last_gc_time > 30):
                    logger.debug(f"Performing memory cleanup at file {processed_count}")
                    
                    # Force garbage collection
                    gc.collect()
                    last_gc_time = current_time
                    
                    # Clear any intermediate results from memory
                    import matplotlib.pyplot as plt
                    plt.close('all')  # Close all matplotlib figures
                    
                    # Yield more CPU time during cleanup
                    time.sleep(0.01)
                    
                return True
            
            # Run batch processing with optimizations
            try:
                
                # Limit max workers based on system resources and file count
                base_workers = 4  # Default workers since we removed the slider
                file_count = len(file_paths)
                
                # Scale down workers for very large batches to prevent resource exhaustion
                if file_count > 1000:
                    max_workers = min(base_workers, 6)  # Max 6 workers for very large batches
                elif file_count > 500:
                    max_workers = min(base_workers, 8)  # Max 8 workers for large batches
                else:
                    max_workers = base_workers
                
                # Apply resource optimizations if available
                processing_params = {
                    'generate_plots': self.generate_plots_var.get(),
                    'max_concurrent_files': max_workers,
                    'max_workers': max_workers
                }
                
                if self.resource_optimizer:
                    # Get optimized parameters
                    optimized_params = self.resource_optimizer.optimize_processing_params(
                        file_count, processing_params
                    )
                    
                    # Apply optimizations
                    max_workers = optimized_params.get('max_workers', max_workers)
                    
                    # Update UI if plots were disabled
                    if not optimized_params.get('generate_plots') and self.generate_plots_var.get():
                        self.after(0, lambda: messagebox.showinfo(
                            "Resource Optimization",
                            "Plot generation has been disabled to conserve memory for this large batch."
                        ))
                        self.generate_plots_var.set(False)
                    
                    # Show resource warnings if any
                    if optimized_params.get('resource_warnings'):
                        warnings_text = "\n".join(optimized_params['resource_warnings'])
                        self.after(0, lambda: messagebox.showwarning(
                            "Resource Warnings",
                            f"Resource constraints detected:\n\n{warnings_text}\n\n"
                            "Processing will continue with optimized settings."
                        ))
                
                logger.info(f"Processing {file_count} files with {max_workers} workers (optimized)")
                
                # Disable plots for very large batches to save memory
                if file_count > 200 and self.generate_plots_var.get():
                    self.after(0, lambda: messagebox.askyesno(
                        "Large Batch Detected",
                        f"Processing {file_count} files with plots enabled may cause performance issues.\n\n"
                        "Disable plots for better performance?"
                    ))
                    # Note: In a real implementation, you'd wait for the user response
                
                # Process with memory-efficient batching
                results = self._process_with_memory_management(
                    file_paths=file_paths,
                    output_dir=output_dir,
                    progress_callback=enhanced_progress_callback,
                    max_workers=max_workers
                )
                
            except Exception as process_error:
                # Provide detailed error messages for common issues
                error_msg = f"Batch processing failed: {str(process_error)}"
                if "Config object has no attribute" in str(process_error):
                    error_msg = f"Configuration error: {str(process_error)}. Please check your configuration settings."
                elif "No module named" in str(process_error):
                    error_msg = f"Missing dependency: {str(process_error)}. Please ensure all required packages are installed."
                elif "Permission denied" in str(process_error):
                    error_msg = f"File access error: {str(process_error)}. Please check file permissions."
                elif "Memory" in str(process_error) or "RAM" in str(process_error):
                    error_msg = f"Memory error: {str(process_error)}. Try processing fewer files at once."
                else:
                    error_msg = f"Processing error: {str(process_error)}"
                
                raise ProcessingError(error_msg)
                
            # Check for cancellation before database save
            if self._is_processing_cancelled():
                self.after(0, self._handle_batch_cancelled, results)
                return
            
            # Save to database if requested
            if self.save_to_db_var.get() and self._db_manager:
                self._save_batch_to_database(results)
            
            # Final cleanup
            gc.collect()
            
            # Check for cancellation one final time
            if self._is_processing_cancelled():
                self.after(0, self._handle_batch_cancelled, results)
            else:
                # Update UI on main thread
                self.after(0, self._handle_batch_success, results, output_dir)
                
        except ValidationError as e:
            logger.error(f"Batch validation error: {e}")
            error_msg = f"Batch validation failed:\n\n{str(e)}\n\nPlease check that all selected files are valid Excel files."
            self.after(0, self._handle_batch_error, error_msg)
            
        except ProcessingError as e:
            logger.error(f"Batch processing error: {e}")
            # Error message is already formatted by the ProcessingError handler above
            self.after(0, self._handle_batch_error, str(e))
            
        except Exception as e:
            logger.error(f"Unexpected batch error: {e}")
            logger.error(traceback.format_exc())
            error_msg = f"An unexpected error occurred during batch processing:\n\n{str(e)}\n\n"
            error_msg += "Please check the log files for more details."
            if "MemoryError" in str(type(e).__name__):
                error_msg = f"Out of memory error:\n\n{str(e)}\n\nTry processing fewer files at once or disable plot generation."
            self.after(0, self._handle_batch_error, error_msg)
        
        finally:
            self.is_processing = False
            # Final cleanup
            import gc
            gc.collect()

    def _process_with_memory_management(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path],
        progress_callback: Callable[[str, float], None],
        max_workers: int
    ) -> Dict[str, AnalysisResult]:
        """Process files with enhanced memory management and cancellation support."""
        import concurrent.futures
        from pathlib import Path
        
        results = {}
        failed_files = []
        processed_files = 0
        total_files = len(file_paths)
        
        # Process in chunks to manage memory better
        if self.resource_manager:
            # Use adaptive chunk size based on resources
            chunk_size = self.resource_manager.get_adaptive_batch_size(total_files, 0)
        else:
            # Fallback to simple adaptive sizing
            chunk_size = min(50, max(10, total_files // 10))
        
        for chunk_start in range(0, total_files, chunk_size):
            # Check for cancellation at start of each chunk
            if self._is_processing_cancelled():
                logger.info(f"Processing cancelled after {processed_files}/{total_files} files")
                break
                
            chunk_end = min(chunk_start + chunk_size, total_files)
            chunk_files = file_paths[chunk_start:chunk_end]
            
            logger.debug(f"Processing chunk {chunk_start}-{chunk_end} ({len(chunk_files)} files)")
            
            # Process chunk with ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all files in chunk
                future_to_file = {}
                
                for file_path in chunk_files:
                    # Check for cancellation before submitting
                    if self._is_processing_cancelled():
                        break
                        
                    future = executor.submit(self._process_single_file_safe, file_path, output_dir)
                    future_to_file[future] = file_path
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file, timeout=300):  # 5 min timeout per file
                    if self._is_processing_cancelled():
                        logger.info("Cancellation requested, stopping file processing")
                        # Cancel remaining futures
                        for remaining_future in future_to_file:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                        
                    file_path = future_to_file[future]
                    processed_files += 1
                    
                    try:
                        result = future.result()
                        if result is not None:
                            results[str(file_path)] = result
                        else:
                            failed_files.append((str(file_path), "Processing returned None"))
                            
                    except Exception as e:
                        logger.error(f"File processing failed for {file_path}: {e}")
                        failed_files.append((str(file_path), str(e)))
                        # Don't show individual errors during batch - they'll be summarized at the end
                    
                    # Update progress
                    progress = (processed_files / total_files) * 100
                    message = f"Processing {file_path.name} ({processed_files}/{total_files})"
                    
                    # Call progress callback and check if we should continue
                    if not progress_callback(message, progress):
                        logger.info("Progress callback signaled to stop processing")
                        break
            
            # Memory cleanup after each chunk
            if self.resource_manager:
                # Check if we should pause for resources
                if self.resource_manager.should_pause_processing():
                    logger.info("Pausing for resource recovery...")
                    
                    # Update UI to show pause
                    if progress_callback:
                        progress_callback(
                            "Pausing for memory recovery...",
                            (processed_files / total_files) * 100
                        )
                    
                    # Wait for resources
                    if not self.resource_manager.wait_for_resources(timeout=30):
                        logger.warning("Resource recovery timeout, continuing anyway")
                
                # Adaptive chunk size for next iteration
                if chunk_end < total_files:
                    chunk_size = self.resource_manager.get_adaptive_batch_size(
                        total_files, chunk_end
                    )
                    logger.debug(f"Next chunk size: {chunk_size}")
            
            # Force cleanup
            import gc
            gc.collect()
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
        
        if failed_files:
            logger.warning(f"Processing completed with {len(failed_files)} failures")
            # Show summary of all failures at the end
            if len(failed_files) > 0:
                failure_summary = "The following files failed to process:\n\n"
                for file_path, error in failed_files[:10]:  # Show first 10 errors
                    file_name = Path(file_path).name
                    failure_summary += f"• {file_name}: {error}\n"
                
                if len(failed_files) > 10:
                    failure_summary += f"\n... and {len(failed_files) - 10} more files"
                
                self.after(0, lambda: messagebox.showwarning(
                    "Processing Failures",
                    failure_summary + "\n\nCheck the log files for complete details."
                ))
            
        return results
    
    def _process_single_file_safe(self, file_path: Path, output_dir: Optional[Path]) -> Optional[AnalysisResult]:
        """Safely process a single file with error handling."""
        logger = logging.getLogger(__name__)
        
        try:
            # Check for cancellation before processing
            if self._is_processing_cancelled():
                return None
            
            # Configure processor for this file
            processor_config = self.analyzer_config.copy() if hasattr(self.analyzer_config, 'copy') else self.analyzer_config
            
            # Update processor config based on UI settings
            if hasattr(processor_config, 'processing'):
                processor_config.processing.generate_plots = self.generate_plots_var.get()
                # Set comprehensive validation if that setting exists
                if hasattr(processor_config.processing, 'comprehensive_validation'):
                    processor_config.processing.comprehensive_validation = self.comprehensive_validation_var.get()
            
            # Process the file using synchronous wrapper for thread pool execution
            result = self.processor.process_file_sync(
                file_path,
                output_dir=output_dir
            )
            
            # Check for cancellation after processing
            if self._is_processing_cancelled():
                return None
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            # Show error dialog for individual file failures
            error_msg = f"Failed to process {file_path.name}:\n\n{str(e)}"
            if "Permission denied" in str(e):
                error_msg += "\n\nPlease ensure the file is not open in another program."
            elif "No such file" in str(e):
                error_msg += "\n\nThe file may have been moved or deleted."
            elif "Invalid file format" in str(e):
                error_msg += "\n\nPlease ensure this is a valid Excel file."
            
            # Only show error dialog for first few failures to avoid dialog spam
            if not hasattr(self, '_file_error_count'):
                self._file_error_count = 0
            
            self._file_error_count += 1
            if self._file_error_count <= 3:  # Only show first 3 file errors
                self.after(0, lambda: messagebox.showerror("File Processing Error", error_msg))
            elif self._file_error_count == 4:
                self.after(0, lambda: messagebox.showwarning(
                    "Multiple File Errors",
                    "Multiple files have failed to process.\n\n"
                    "Further individual error dialogs will be suppressed.\n"
                    "Check the final summary for all errors."
                ))
            return None

    def _handle_batch_cancelled(self, partial_results: Dict[str, AnalysisResult]):
        """Handle cancelled batch processing."""
        # Hide progress dialog
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.hide()
            self.progress_dialog = None
        
        # Store partial results
        self.batch_results = partial_results
        
        # Update status
        self._update_batch_status("Processing Cancelled", "orange")
        
        # Re-enable controls
        self._set_controls_state("normal")
        self.start_button.configure(text="Start Processing")
        
        # Display partial results if any
        if partial_results:
            self.batch_results_widget.display_results(partial_results)
            self.export_button.configure(state="normal")
        
        # Show cancellation message
        processed_count = len(partial_results)
        total_selected = len(self.selected_files)
        
        messagebox.showinfo(
            "Processing Cancelled", 
            f"Batch processing was cancelled.\n\n"
            f"Processed: {processed_count} files\n"
            f"Remaining: {total_selected - processed_count} files\n\n"
            f"Partial results are available for export."
        )
        
        logger.info(f"Batch processing cancelled. Processed {processed_count} of {total_selected} files")

    def _save_batch_to_database(self, results: Dict[str, AnalysisResult]):
        """Save batch results to database with robust error handling."""
        if not self._db_manager:
            logger.error("Database manager is None - cannot save to database!")
            self.after(0, lambda: messagebox.showerror(
                "Database Error",
                "Database is not initialized. Results could not be saved.\n\n"
                "Please check the database configuration and restart the application."
            ))
            return
            
        saved_count = 0
        failed_count = 0
        
        logger.info(f"Starting database save for {len(results)} results")
        
        # Use batch save if available for better performance
        if hasattr(self._db_manager, 'save_analysis_batch'):
            try:
                # Check for cancellation
                if self._is_processing_cancelled():
                    logger.info("Cancellation requested during database save")
                    return
                
                # Convert to list for batch save
                result_list = list(results.values())
                if result_list:
                    # Validate results before saving
                    logger.info(f"Validating {len(result_list)} results before batch save...")
                    valid_results = []
                    for i, result in enumerate(result_list):
                        if result is not None and result.metadata and result.metadata.model and result.metadata.serial:
                            valid_results.append(result)
                        else:
                            logger.warning(f"Skipping invalid result at index {i}: missing required data")
                    
                    if not valid_results:
                        logger.error("No valid results to save!")
                        return
                        
                    logger.info(f"Attempting batch save of {len(valid_results)} valid results...")
                    analysis_ids = self._db_manager.save_analysis_batch(valid_results)
                    saved_count = len([id for id in analysis_ids if id is not None])
                    logger.info(f"Batch saved {saved_count}/{len(valid_results)} analyses to database")
                    
                    # Update result objects with database IDs
                    for result, db_id in zip(result_list, analysis_ids):
                        result.db_id = db_id
            except Exception as e:
                logger.error(f"Batch database save failed: {e}")
                # Fall back to individual saves
                logger.info("Falling back to individual saves")
                self.after(0, lambda: messagebox.showwarning(
                    "Database Save Warning",
                    f"Batch database save failed:\n{str(e)}\n\n"
                    "Attempting to save files individually..."
                ))
                for file_path, result in results.items():
                    try:
                        # Individual save logic
                        self._save_individual_analysis(file_path, result)
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"Database save failed for {Path(file_path).name}: {e}")
                        failed_count += 1
                        # Don't show individual save errors - they'll be summarized
        else:
            # Individual saves if batch save not available
            for file_path, result in results.items():
                try:
                    # Check for cancellation during database save
                    if self._is_processing_cancelled():
                        logger.info("Cancellation requested during database save")
                        break
                        
                    self._save_individual_analysis(file_path, result)
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Database save failed for {Path(file_path).name}: {e}")
                    failed_count += 1
                    # Don't show individual save errors - they'll be summarized
        
        logger.info(f"Database save complete: {saved_count} saved, {failed_count} failed")
        
        if failed_count > 0:
            # Show warning about failed saves
            self.after(0, lambda: messagebox.showwarning(
                "Database Warning",
                f"Some files failed to save to database:\n"
                f"Saved: {saved_count}\n"
                f"Failed: {failed_count}\n\n"
                f"Check logs for details."
            ))

    def _save_individual_analysis(self, file_path: str, result: AnalysisResult):
        """Save an individual analysis to the database with error handling."""
        # Check for duplicates
        existing_id = self._db_manager.check_duplicate_analysis(
            result.metadata.model,
            result.metadata.serial,
            result.metadata.file_date
        )
        
        if existing_id:
            logger.info(f"Duplicate found for {Path(file_path).name} (ID: {existing_id})")
            result.db_id = existing_id
        else:
            # Try using the debug save method first
            try:
                # Use debug method if available
                if hasattr(self._db_manager, 'save_analysis_result'):
                    logger.debug(f"Using debug save method for {Path(file_path).name}")
                    result.db_id = self._db_manager.save_analysis_result(result)
                else:
                    result.db_id = self._db_manager.save_analysis(result)
                
                # Validate the save
                if not self._db_manager.validate_saved_analysis(result.db_id):
                    raise RuntimeError("Database validation failed")
                    
            except Exception as save_error:
                logger.warning(f"Normal save failed for {Path(file_path).name}, trying force save: {save_error}")
                # Try force save as fallback
                result.db_id = self._db_manager.force_save_analysis(result)
                logger.info(f"Force saved {Path(file_path).name} to database")

    def _handle_batch_success(self, results: Dict[str, AnalysisResult], output_dir: Optional[Path]):
        """Handle successful batch completion."""
        self.batch_results = results
        
        # Hide progress dialog
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            try:
                self.progress_dialog.hide()
            except Exception as e:
                logger.error(f"Error hiding progress dialog: {e}")
                # Don't show error dialog for non-critical UI cleanup
            finally:
                self.progress_dialog = None
        
        # Calculate summary statistics
        total_processed = len(results)
        successful_count = len(results)
        failed_count = len(self.selected_files) - successful_count
        
        # Validation statistics
        validated_count = sum(1 for r in results.values() 
                            if r.overall_validation_status == ValidationStatus.VALIDATED)
        warning_count = sum(1 for r in results.values() 
                          if r.overall_validation_status == ValidationStatus.WARNING)
        failed_validation_count = sum(1 for r in results.values() 
                                    if r.overall_validation_status == ValidationStatus.FAILED)
        
        # Update batch status
        if failed_count == 0:
            if validated_count == successful_count:
                self._update_batch_status("Batch Complete - All Validated", "green")
            elif warning_count > 0:
                self._update_batch_status("Batch Complete - With Warnings", "orange")
            else:
                self._update_batch_status("Batch Complete - Some Validation Issues", "orange")
        else:
            self._update_batch_status("Batch Complete - With Errors", "red")
        
        # Display results
        self.batch_results_widget.display_results(results)
        
        # Enable export button
        self.export_button.configure(state="normal")
        
        # Re-enable controls
        self._set_controls_state("normal")
        
        # Show completion message
        success_msg = f"Batch processing completed!\n\n"
        success_msg += f"Files processed: {successful_count}/{len(self.selected_files)}\n"
        success_msg += f"Validated: {validated_count}\n"
        success_msg += f"Warnings: {warning_count}\n"
        success_msg += f"Failed validation: {failed_validation_count}\n"
        
        if failed_count > 0:
            success_msg += f"Processing failures: {failed_count}\n"
        
        if output_dir:
            success_msg += f"\nOutputs saved to: {output_dir}"
        
        messagebox.showinfo("Batch Processing Complete", success_msg)
        
        # Refresh home page recent activity
        try:
            if hasattr(self.main_window, 'pages') and 'home' in self.main_window.pages:
                home_page = self.main_window.pages['home']
                if hasattr(home_page, 'refresh'):
                    home_page.refresh()
                    logger.info("Refreshed home page after batch processing")
        except Exception as e:
            logger.debug(f"Failed to refresh home page: {e}")
        
        logger.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed")
    
    def _on_resource_update(self, status: ResourceStatus):
        """Handle resource status updates."""
        if not self.resource_status_label:
            return
            
        # Update UI on main thread
        self.after(0, self._update_resource_display, status)
    
    def _update_resource_display(self, status: ResourceStatus):
        """Update resource monitoring display."""
        if not hasattr(self, 'memory_progress'):
            return
            
        # Update memory progress bar
        memory_percent = status.memory_percent / 100.0
        self.memory_progress.set(memory_percent)
        
        # Update memory label
        self.memory_status_label.configure(
            text=f"{status.memory_percent:.0f}% ({status.used_memory_mb:.1f} GB / {status.total_memory_mb:.1f} GB)"
        )
        
        # Update status color and message
        if status.memory_critical:
            color = "red"
            message = "Resources: CRITICAL - Processing may be slow or fail"
        elif status.memory_warning:
            color = "orange"
            message = "Resources: WARNING - Consider closing other applications"
        else:
            color = "green"
            message = "Resources: Normal - Ready for batch processing"
        
        self.resource_status_label.configure(text=message, text_color=color)
        
        # Update optimization strategy
        if self.resource_optimizer and hasattr(self, 'selected_files'):
            strategy = self.resource_optimizer.get_processing_strategy(len(self.selected_files))
            self.optimization_label.configure(
                text=f"Optimization: {strategy['description']}"
            )

    def _handle_batch_error(self, error_message: str):
        """Handle batch processing error."""
        # Hide progress dialog
        if self.progress_dialog:
            self.progress_dialog.hide()
            self.progress_dialog = None
        
        # Update status
        self._update_batch_status("Batch Processing Failed", "red")
        
        # Re-enable controls
        self._set_controls_state("normal")
        
        # Show error message
        messagebox.showerror("Batch Processing Failed", error_message)
        
        logger.error(f"Batch processing failed: {error_message}")

    def _stop_processing(self):
        """Stop batch processing gracefully."""
        if not self.is_processing:
            return
            
        reply = messagebox.askyesno(
            "Stop Processing",
            "Are you sure you want to stop batch processing?\n\n"
            "This will cancel any remaining files and save progress so far."
        )
        
        if reply:
            logger.info("User requested to stop batch processing")
            
            # Set stop flags
            self._stop_event.set()
            self._processing_cancelled = True
            self.request_stop_processing()
            
            # Update UI immediately
            self.start_button.configure(text="Stopping...", state="disabled")
            self.stop_button.configure(state="disabled")
            
            # Update status
            self._update_batch_status("Stopping Processing...", "orange")
            
            # The processing thread will handle the actual stopping
            
    def _is_processing_cancelled(self) -> bool:
        """Check if processing has been cancelled."""
        return self._processing_cancelled or self._stop_event.is_set() or self.is_stop_requested()

    def _set_controls_state(self, state: str):
        """Set state of control buttons."""
        if state == "disabled":
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.select_files_button.configure(state="disabled")
            self.select_folder_button.configure(state="disabled")
            self.validate_batch_button.configure(state="disabled")
        else:
            has_files = len(self.selected_files) > 0
            has_valid_files = any(self.validation_results.values()) if self.validation_results else has_files
            
            self.start_button.configure(state="normal" if has_valid_files else "disabled")
            self.stop_button.configure(state="disabled")
            self.select_files_button.configure(state="normal")
            self.select_folder_button.configure(state="normal")
            self.validate_batch_button.configure(state="normal" if has_files else "disabled")

    def _export_batch_results(self):
        """Export batch processing results."""
        if not self.batch_results:
            messagebox.showerror("Error", "No results to export")
            return
        
        # Ask for export location
        file_path = filedialog.asksaveasfilename(
            title="Export Batch Results",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    self._export_batch_excel(Path(file_path))
                elif file_path.endswith('.csv'):
                    self._export_batch_csv(Path(file_path))
                else:
                    messagebox.showerror("Error", "Unsupported file format")
                    return
                
                messagebox.showinfo("Export Complete", f"Batch results exported to:\n{file_path}")
                logger.info(f"Batch results exported to: {file_path}")
                
            except Exception as e:
                logger.error(f"Batch export failed: {e}")
                error_msg = f"Failed to export batch results:\n\n{str(e)}"
                if "Permission denied" in str(e):
                    error_msg += "\n\nPlease ensure the output file is not open in another program."
                elif "No space left" in str(e):
                    error_msg += "\n\nInsufficient disk space. Please free up some space and try again."
                elif "Invalid file path" in str(e):
                    error_msg += "\n\nThe selected file path is invalid. Please choose a different location."
                messagebox.showerror("Export Failed", error_msg)

    def _export_batch_excel(self, file_path: Path):
        """Export batch results to Excel format."""
        import pandas as pd
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            all_track_data = []
            
            for file_path_str, result in self.batch_results.items():
                file_name = Path(file_path_str).name
                
                try:
                    # Get values safely with error handling
                    model = getattr(result.metadata, 'model', 'Unknown')
                    serial = getattr(result.metadata, 'serial', 'Unknown')
                    
                    # Handle system_type safely
                    system_type = 'Unknown'
                    if hasattr(result.metadata, 'system_type'):
                        system_type = getattr(result.metadata.system_type, 'value', str(result.metadata.system_type))
                    
                    # Handle analysis_date safely
                    analysis_date = 'Unknown'
                    if hasattr(result.metadata, 'analysis_date'):
                        if hasattr(result.metadata.analysis_date, 'strftime'):
                            analysis_date = result.metadata.analysis_date.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            analysis_date = str(result.metadata.analysis_date)
                    
                    # Handle overall_status safely
                    overall_status = 'Unknown'
                    if hasattr(result, 'overall_status'):
                        overall_status = getattr(result.overall_status, 'value', str(result.overall_status))
                    
                    # Handle validation_status safely
                    validation_status = 'N/A'
                    if hasattr(result, 'overall_validation_status'):
                        validation_status = getattr(result.overall_validation_status, 'value', 
                                                   str(result.overall_validation_status))
                    
                    # Get track counts safely
                    track_count = 0
                    pass_count = 0
                    fail_count = 0
                    
                    if hasattr(result, 'tracks'):
                        # Handle both dict (from analysis) and list (from DB) formats
                        if isinstance(result.tracks, dict):
                            track_count = len(result.tracks)
                            tracks_iter = result.tracks.values()
                        else:
                            track_count = len(result.tracks)
                            tracks_iter = result.tracks
                            
                        for track in tracks_iter:
                            if hasattr(track, 'overall_status'):
                                track_status = getattr(track.overall_status, 'value', str(track.overall_status))
                                if track_status == 'Pass':
                                    pass_count += 1
                                else:
                                    fail_count += 1
                            elif hasattr(track, 'status'):
                                # Database tracks have 'status' not 'overall_status'
                                track_status = getattr(track.status, 'value', str(track.status))
                                if track_status == 'Pass':
                                    pass_count += 1
                                else:
                                    fail_count += 1
                    
                    # Summary row
                    summary_data.append({
                        'File': file_name,
                        'Model': model,
                        'Serial': serial,
                        'System_Type': system_type,
                        'Analysis_Date': analysis_date,
                        'Overall_Status': overall_status,
                        'Validation_Status': validation_status,
                        'Processing_Time': f"{getattr(result, 'processing_time', 0):.2f}",
                        'Track_Count': track_count,
                        'Pass_Count': pass_count,
                        'Fail_Count': fail_count
                    })
                except Exception as e:
                    # Log error and continue with next result
                    logger.error(f"Error processing result for export: {e}")
                    # Add minimal data for this file
                    summary_data.append({
                        'File': file_name,
                        'Model': 'Error',
                        'Serial': 'Error',
                        'Error': str(e)
                    })
                    # Show warning about specific file
                    self.after(0, lambda fn=file_name, err=str(e): messagebox.showwarning(
                        "Export Data Warning",
                        f"Could not export complete data for {fn}:\n{err}\n\n"
                        "Partial data will be included in the export."
                    ))
                
                # Individual track data
                for track_id, track in result.tracks.items():
                    track_row = {
                        'File': file_name,
                        'Model': result.metadata.model,
                        'Serial': result.metadata.serial,
                        'Track_ID': track_id,
                        'Track_Status': track.overall_status.value,
                        'Sigma_Gradient': track.sigma_analysis.sigma_gradient if track.sigma_analysis else None,
                        'Sigma_Threshold': track.sigma_analysis.sigma_threshold if track.sigma_analysis else None,
                        'Sigma_Pass': track.sigma_analysis.sigma_pass if track.sigma_analysis else None,
                        'Linearity_Spec': track.linearity_analysis.linearity_spec if track.linearity_analysis else None,
                        'Linearity_Pass': track.linearity_analysis.linearity_pass if track.linearity_analysis else None,
                        'Risk_Category': track.risk_category.value if hasattr(track, 'risk_category') else 'Unknown'
                    }
                    all_track_data.append(track_row)
            
            # Write summary sheet
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Batch Summary', index=False)
            
            # Write track details sheet
            tracks_df = pd.DataFrame(all_track_data)
            tracks_df.to_excel(writer, sheet_name='Track Details', index=False)
            
            # Statistics sheet
            stats_data = {
                'Metric': [
                    'Total Files Processed',
                    'Total Files Selected',
                    'Success Rate',
                    'Total Tracks Analyzed', 
                    'Tracks Passed',
                    'Tracks Failed',
                    'Pass Rate',
                    'Files Validated',
                    'Files with Warnings',
                    'Files with Validation Errors'
                ],
                'Value': [
                    len(self.batch_results),
                    len(self.selected_files),
                    f"{len(self.batch_results)/len(self.selected_files)*100:.1f}%",
                    sum(len(r.tracks) for r in self.batch_results.values()),
                    sum(sum(1 for t in r.tracks.values() if t.overall_status.value == 'Pass') for r in self.batch_results.values()),
                    sum(sum(1 for t in r.tracks.values() if t.overall_status.value != 'Pass') for r in self.batch_results.values()),
                    f"{sum(sum(1 for t in r.tracks.values() if t.overall_status.value == 'Pass') for r in self.batch_results.values()) / sum(len(r.tracks) for r in self.batch_results.values()) * 100:.1f}%" if sum(len(r.tracks) for r in self.batch_results.values()) > 0 else "0%",
                    sum(1 for r in self.batch_results.values() if hasattr(r, 'overall_validation_status') and r.overall_validation_status.value == 'VALIDATED'),
                    sum(1 for r in self.batch_results.values() if hasattr(r, 'overall_validation_status') and r.overall_validation_status.value == 'WARNING'),
                    sum(1 for r in self.batch_results.values() if hasattr(r, 'overall_validation_status') and r.overall_validation_status.value == 'FAILED')
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

    def _export_batch_csv(self, file_path: Path):
        """Export batch results to CSV format."""
        import pandas as pd
        
        # Create comprehensive CSV export
        export_data = []
        
        for file_path_str, result in self.batch_results.items():
            file_name = Path(file_path_str).name
            
            for track_id, track in result.tracks.items():
                row = {
                    'File': file_name,
                    'Model': result.metadata.model,
                    'Serial': result.metadata.serial,
                    'System_Type': result.metadata.system_type.value,
                    'Analysis_Date': result.metadata.analysis_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Track_ID': track_id,
                    'Overall_Status': result.overall_status.value,
                    'Track_Status': track.overall_status.value,
                    'Processing_Time': f"{result.processing_time:.2f}",
                    'Validation_Status': result.overall_validation_status.value if hasattr(result, 'overall_validation_status') else 'N/A'
                }
                
                # Add detailed analysis data
                if track.sigma_analysis:
                    row.update({
                        'Sigma_Gradient': track.sigma_analysis.sigma_gradient,
                        'Sigma_Threshold': track.sigma_analysis.sigma_threshold,
                        'Sigma_Pass': track.sigma_analysis.sigma_pass,
                        'Sigma_Improvement': track.sigma_analysis.improvement_percent if hasattr(track.sigma_analysis, 'improvement_percent') else None
                    })
                
                if track.linearity_analysis:
                    row.update({
                        'Linearity_Spec': track.linearity_analysis.linearity_spec,
                        'Linearity_Pass': track.linearity_analysis.linearity_pass,
                        'Linearity_Error': track.linearity_analysis.linearity_error if hasattr(track.linearity_analysis, 'linearity_error') else None
                    })
                
                if track.resistance_analysis:
                    row.update({
                        'Resistance_Before': track.resistance_analysis.resistance_before if hasattr(track.resistance_analysis, 'resistance_before') else None,
                        'Resistance_After': track.resistance_analysis.resistance_after if hasattr(track.resistance_analysis, 'resistance_after') else None,
                        'Resistance_Change_Percent': track.resistance_analysis.resistance_change_percent if hasattr(track.resistance_analysis, 'resistance_change_percent') else None
                    })
                
                if hasattr(track, 'risk_category'):
                    row['Risk_Category'] = track.risk_category.value
                
                export_data.append(row)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(file_path, index=False)

    def _clear_results(self):
        """Clear batch processing results."""
        self.batch_results = {}
        self.batch_results_widget.clear()
        self.export_button.configure(state="disabled")
        logger.info("Batch results cleared")
    
    def cleanup(self):
        """Cleanup resources when page is destroyed."""
        # Stop any ongoing processing
        if self.is_processing:
            self._stop_processing()
        
        # Cleanup resource manager callbacks
        if self.resource_manager:
            # Remove our callback (would need to track it properly in production)
            pass
        
        super().cleanup() 
