"""
Batch Processing Page for Laser Trim Analyzer

Handles batch analysis of multiple Excel files with comprehensive
validation, progress tracking, and responsive design.
"""

# import asyncio  # Not used, commented out
import gc
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk

from laser_trim_analyzer.core.config import get_config
from laser_trim_analyzer.core.exceptions import ProcessingError, ValidationError
from laser_trim_analyzer.core.models import AnalysisResult, AnalysisStatus, ValidationStatus
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.core.large_scale_processor import LargeScaleProcessor
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.utils.file_utils import ensure_directory
from laser_trim_analyzer.utils.report_generator import ReportGenerator
from laser_trim_analyzer.utils.batch_logging import setup_batch_logging, BatchProcessingLogger

# Set up logging before using it
logger = logging.getLogger(__name__)

from laser_trim_analyzer.gui.widgets.metric_card_ctk import MetricCard
from laser_trim_analyzer.gui.widgets.batch_results_widget_ctk import BatchResultsWidget
from laser_trim_analyzer.gui.widgets.progress_widgets_ctk import BatchProgressDialog
from laser_trim_analyzer.gui.widgets.hover_fix import fix_hover_glitches, stabilize_layout

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
    # Create a placeholder type for type annotations
    from typing import Any
    ResourceStatus = Any


class BatchProcessingPage(ctk.CTkFrame):
    """Batch processing page with comprehensive validation and responsive design."""

    def __init__(self, parent, main_window, **kwargs):
        """Initialize batch processing page with configuration validation."""
        # Initialize as CTkFrame to avoid widget hierarchy issues
        super().__init__(parent, **kwargs)
        self.main_window = main_window
        
        # Add missing BasePage functionality
        self.is_visible = False
        self.needs_refresh = True
        self._stop_requested = False
        self._shutting_down = False  # Prevent callback errors during shutdown
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Add missing methods that are called
        self.is_stop_requested = lambda: self._stop_requested
        self.request_stop_processing = lambda: setattr(self, '_stop_requested', True)
        
        # Initialize responsive layout attributes
        self.current_size_class = 'large'  # Default size class
        
        # Initialize batch logger (will be created when batch starts)
        self.batch_logger: Optional[BatchProcessingLogger] = None
        
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
        self.last_output_dir: Optional[Path] = None
        self.failed_files: List[Tuple[str, str]] = []  # Track failed files and errors
        
        # Validate required directories exist
        self._ensure_required_directories()
        
        # Processing control with thread safety
        self._stop_event = threading.Event()
        self._processing_cancelled = False
        self._state_lock = threading.Lock()  # Lock for thread-safe state access
        
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
        
        # Initialize report generator
        self.report_generator = ReportGenerator()
        
        logger.info(f"Batch processing page initialized (CTK widgets: Base={USING_CTK_BASE}, Metric={USING_CTK_METRIC}, BatchResults={USING_CTK_BATCH_RESULTS}, Progress={USING_CTK_PROGRESS})")
        
        # Initialize the page
        try:
            self._create_page()
        except Exception as e:
            self.logger.error(f"Error creating page: {e}", exc_info=True)
            self._create_error_page(str(e))

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
        
        # Apply hover fixes after page creation
        self._safe_after(100, self._apply_hover_fixes)

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

    def _apply_hover_fixes(self):
        """Apply hover fixes to prevent glitching and shifting."""
        try:
            # Check if widget is ready
            if not self.winfo_exists():
                return
                
            # Fix hover glitches on all widgets
            fix_hover_glitches(self)
            
            # Stabilize layout to prevent shifting if container exists
            if hasattr(self, 'main_container') and self.main_container.winfo_exists():
                stabilize_layout(self.main_container)
            
            logger.debug("Hover fixes applied successfully")
        except Exception as e:
            # Only log debug level since this is not critical
            logger.debug(f"Could not apply hover fixes: {e}")
    
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
        
        # File selection instruction label
        instruction_label = ctk.CTkLabel(
            self.file_selection_frame,
            text="Use the buttons below to select Excel files for batch processing",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        instruction_label.pack(pady=(5, 10), padx=15)
        
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
        
        self.force_reprocess_var = ctk.BooleanVar(value=False)
        self.force_reprocess_check = ctk.CTkCheckBox(
            self.options_container,
            text="Force Reprocess (Skip Duplicate Check)",
            variable=self.force_reprocess_var
        )

        # Incremental processing option - skip already-processed files (Phase 1 Day 2)
        self.skip_processed_var = ctk.BooleanVar(value=True)  # Default ON for faster processing
        self.skip_processed_check = ctk.CTkCheckBox(
            self.options_container,
            text="Skip Already Processed (Incremental Mode)",
            variable=self.skip_processed_var
        )

        # Store options for responsive layout
        self.option_widgets = [
            self.generate_plots_check,
            self.save_to_db_check,
            self.comprehensive_validation_check,
            self.force_reprocess_check,
            self.skip_processed_check
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
        
        # Master summary panel - shows overall statistics after processing
        self.summary_frame = ctk.CTkFrame(self.results_frame)
        self.summary_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        # Summary will be populated after batch processing completes
        self.summary_label = ctk.CTkLabel(
            self.summary_frame,
            text="Process files to see summary statistics",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.summary_label.pack(pady=20)
        
        # Results widget
        self.batch_results_widget = BatchResultsWidget(self.results_frame)
        self.batch_results_widget.pack(fill='both', expand=True, padx=15, pady=(0, 10))
        
        # Export controls frame
        export_controls_frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        export_controls_frame.pack(pady=(0, 15))
        
        # Export options frame
        export_options_frame = ctk.CTkFrame(export_controls_frame, fg_color="transparent")
        export_options_frame.pack(side='left', padx=(0, 20))
        
        # Include raw data checkbox
        self.include_raw_data_var = ctk.BooleanVar(value=False)
        self.include_raw_data_check = ctk.CTkCheckBox(
            export_options_frame,
            text="Include Raw Data (up to 10 files)",
            variable=self.include_raw_data_var
        )
        self.include_raw_data_check.pack(side='left', padx=5)
        
        # Export buttons frame
        export_buttons_frame = ctk.CTkFrame(export_controls_frame, fg_color="transparent")
        export_buttons_frame.pack(side='left')
        
        # Export Excel button
        self.export_excel_button = ctk.CTkButton(
            export_buttons_frame,
            text="Export to Excel",
            command=lambda: self._export_batch_results('excel'),
            state="disabled"
        )
        self.export_excel_button.pack(side='left', padx=5)
        
        # Export HTML button
        self.export_html_button = ctk.CTkButton(
            export_buttons_frame,
            text="Export to HTML",
            command=lambda: self._export_batch_results('html'),
            state="disabled"
        )
        self.export_html_button.pack(side='left', padx=5)
        
        # Export CSV button (legacy)
        self.export_csv_button = ctk.CTkButton(
            export_buttons_frame,
            text="Export to CSV",
            command=lambda: self._export_batch_results('csv'),
            state="disabled"
        )
        self.export_csv_button.pack(side='left', padx=5)
        
        # Output folder button
        self.output_folder_button = ctk.CTkButton(
            export_buttons_frame,
            text="Open Output Folder",
            command=self._open_output_folder,
            state="disabled"
        )
        self.output_folder_button.pack(side='left', padx=5)

    def _setup_responsive_layout(self):
        """Setup initial responsive layout."""
        self._arrange_responsive_elements()
        
    def _handle_responsive_layout(self, size_class: str):
        """Handle responsive layout changes."""
        # CTkFrame doesn't have _handle_responsive_layout, so don't call super
        # super()._handle_responsive_layout(size_class)
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
            
        # Simple responsive logic for validation cards
        num_cards = len(self.validation_cards)
        if num_cards <= 2:
            columns = 1
        elif num_cards <= 4:
            columns = 2
        else:
            columns = 3
        
        for i, card in enumerate(self.validation_cards):
            row = i // columns
            col = i % columns
            # Simple padding instead of responsive padding
            card.grid(row=row, column=col, sticky='ew', padx=5, pady=5)
            
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
                            self._safe_after(0, lambda count=total_checked: 
                                     self._update_batch_status(f"Scanning... ({count} files found)", "orange"))
                
                # Update UI on main thread
                self._safe_after(0, lambda: self._handle_folder_discovery_complete(excel_files, folder_path))
                
            except Exception as e:
                logger.error(f"Folder discovery failed: {e}")
                self._safe_after(0, lambda: self._handle_folder_discovery_error(str(e)))
        
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
        
        # Clear the drop zone if it exists
        if hasattr(self, 'drop_zone'):
            try:
                self.drop_zone.clear_files()
            except Exception as e:
                logger.debug(f"Could not clear drop zone: {e}")
        
        # Use _set_controls_state to properly disable all buttons when no files
        self._set_controls_state("normal")  # This will disable start button due to no files

    def _update_file_display(self):
        """Update file list display."""
        if self._shutting_down:
            return
        
        try:
            self.file_list_label.configure(text=f"Selected Files ({len(self.selected_files)}):")
            self.file_listbox.configure(state="normal")
        except tk.TclError:
            return  # Widget destroyed
        try:
            self.file_listbox.delete("1.0", tk.END)
            
            for i, file_path in enumerate(self.selected_files, 1):
                validation_status = ""
                if str(file_path) in self.validation_results:
                    status = "✓" if self.validation_results[str(file_path)] else "✗"
                    validation_status = f" [{status}]"
                
                self.file_listbox.insert(tk.END, f"{i:3d}. {file_path.name}{validation_status}\n")
            
            self.file_listbox.configure(state="disabled")
            
            # Enable/disable buttons
            has_files = len(self.selected_files) > 0
            self.validate_batch_button.configure(state="normal" if has_files else "disabled")
        except tk.TclError:
            return  # Widget destroyed

    def _validate_batch(self):
        """Validate all selected files."""
        if not self.selected_files:
            messagebox.showerror("Error", "No files selected")
            return
        
        self._update_batch_status("Validating Batch...", "orange")
        
        # Add progress tracking for validation
        self.validation_progress = 0
        self.validation_total = len(self.selected_files)
        
        # Disable controls during validation
        self._set_controls_state("disabled")
        
        # Show validation progress dialog
        self.validation_progress_dialog = BatchProgressDialog(
            self,
            title="Batch Validation",
            total_files=len(self.selected_files)
        )
        self.validation_progress_dialog.show()
        
        # Update progress bar during validation
        def update_validation_progress():
            if hasattr(self, 'validation_progress') and hasattr(self, 'validation_progress_dialog'):
                progress = self.validation_progress / self.validation_total if self.validation_total > 0 else 0
                status_text = f"Validating file {self.validation_progress} of {self.validation_total}"
                
                # Update both the main progress bar and the dialog
                self.progress_bar.set(progress)
                self._update_batch_status(status_text, "orange")
                
                if self.validation_progress_dialog:
                    self.validation_progress_dialog.update_progress(status_text, progress)
                
                if self.validation_progress < self.validation_total:
                    self._safe_after(500, update_validation_progress)  # Update every 500ms
        
        # Start progress updates
        self._safe_after(100, update_validation_progress)
        
        # Run validation in thread
        def validate():
            try:
                from laser_trim_analyzer.utils.validators import BatchValidator
                
                # For validation, allow much larger batches than processing
                # Validation is just checking files, not processing them
                validation_max_size = max(10000, len(self.selected_files))
                
                # Create progress callback
                def validation_progress_callback(current_file_index):
                    self.validation_progress = current_file_index + 1
                    # Update progress bar immediately during validation
                    progress = self.validation_progress / self.validation_total if self.validation_total > 0 else 0
                    progress = min(progress, 1.0)  # Ensure never exceeds 100%
                    if hasattr(self, 'validation_progress_dialog') and self.validation_progress_dialog:
                        self._safe_after(0, lambda: self.validation_progress_dialog.update_progress(
                            f"Validating file {self.validation_progress} of {self.validation_total}", progress))
                
                validation_result = BatchValidator.validate_batch(
                    file_paths=self.selected_files,
                    max_batch_size=validation_max_size,
                    progress_callback=validation_progress_callback
                )
                
                # Store individual file validation results
                invalid_files = validation_result.metadata.get('invalid_files', [])
                self.validation_results = {}
                
                for file_path in self.selected_files:
                    is_valid = not any(str(file_path) in invalid['file'] for invalid in invalid_files)
                    self.validation_results[str(file_path)] = is_valid
                
                # Update UI on main thread
                self._safe_after(0, lambda: self._handle_batch_validation_result(validation_result))
                
            except Exception as e:
                logger.error(f"Batch validation failed: {e}")
                error_msg = str(e)
                if "No module named" in error_msg:
                    error_msg = f"Missing dependency for validation: {error_msg}\nPlease ensure all required packages are installed."
                elif "Config object has no attribute" in error_msg:
                    error_msg = f"Configuration error: {error_msg}\nPlease check your configuration settings."
                self._safe_after(0, lambda: self._handle_batch_validation_error(error_msg))
        
        thread = threading.Thread(target=validate, daemon=True)
        thread.start()

    def _handle_batch_validation_result(self, validation_result):
        """Handle batch validation result."""
        # Hide validation progress dialog
        if hasattr(self, 'validation_progress_dialog') and self.validation_progress_dialog:
            self.validation_progress_dialog.hide()
            self.validation_progress_dialog = None
        
        # Re-enable controls
        self._set_controls_state("normal")
        
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
        # Hide validation progress dialog
        if hasattr(self, 'validation_progress_dialog') and self.validation_progress_dialog:
            self.validation_progress_dialog.hide()
            self.validation_progress_dialog = None
        
        # Re-enable controls
        self._set_controls_state("normal")
        
        self._update_batch_status("Validation Error", "red")
        messagebox.showerror("Validation Error", f"Batch validation failed:\n{error_message}")

    def _update_batch_status(self, status: str, color: str):
        """Update batch status indicator."""
        if self._shutting_down:
            return
        try:
            self.batch_status_label.configure(text=f"Batch Status: {status}")
        except tk.TclError:
            return  # Widget destroyed
        
        color_map = {
            "green": "#00ff00",
            "orange": "#ffa500",
            "red": "#ff0000", 
            "gray": "#808080"
        }
        
        try:
            self.batch_indicator.configure(text_color=color_map.get(color, "#808080"))
        except tk.TclError:
            return  # Widget destroyed

    def _start_processing(self):
        """Start batch processing."""
        if not self.selected_files:
            messagebox.showerror("Error", "No files selected")
            return
        
        with self._state_lock:
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

        # Incremental processing: Skip already-processed files (Phase 1 Day 2)
        skipped_count = 0
        if self.skip_processed_var.get() and self._db_manager:
            try:
                original_count = len(processable_files)
                processable_files = self._db_manager.get_unprocessed_files(processable_files)
                skipped_count = original_count - len(processable_files)
                if skipped_count > 0:
                    logger.info(f"Incremental mode: Skipping {skipped_count} already-processed files")
                    # Show info message if all files already processed
                    if len(processable_files) == 0:
                        messagebox.showinfo(
                            "All Files Already Processed",
                            f"All {original_count} selected files have already been processed.\n\n"
                            "Uncheck 'Skip Already Processed' to reprocess them."
                        )
                        return
            except Exception as e:
                logger.warning(f"Could not filter already-processed files: {e}")
                # Continue with all files if filter fails

        if not processable_files:
            messagebox.showerror("Error", "No valid files to process")
            return

        # Clear previous results
        self._clear_results()
        
        # Reset error tracking
        self._file_error_count = 0
        
        # Disable controls
        self._set_controls_state("disabled")
        
        # Show progress dialog with skipped count info
        title = "Batch Processing"
        if skipped_count > 0:
            title = f"Batch Processing ({skipped_count} skipped)"
        self.progress_dialog = BatchProgressDialog(
            self,
            title=title,
            total_files=len(processable_files)
        )
        self.progress_dialog.show()

        # Start processing in thread (thread-safe)
        with self._state_lock:
            self.is_processing = True

        # Track processing start time for summary
        self.processing_start_time = time.time()
        # Store skipped count for summary
        self._skipped_processed_count = skipped_count

        self.processing_thread = threading.Thread(
            target=self._run_batch_processing,
            args=(processable_files,),
            daemon=True
        )
        self.processing_thread.start()

        if skipped_count > 0:
            logger.info(f"Started batch processing of {len(processable_files)} files ({skipped_count} already-processed files skipped)")
        else:
            logger.info(f"Started batch processing of {len(processable_files)} files")

    def _run_batch_processing(self, file_paths: List[Path]):
        """Run batch processing in background thread with performance optimizations and stop handling."""
        import gc
        import time
        
        try:
            # Reset stop flags (thread-safe)
            self._stop_event.clear()
            with self._state_lock:
                self._processing_cancelled = False
            self._stop_requested = False  # Reset stop flag directly
            
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
                # Store output dir immediately in case of cancellation
                self.last_output_dir = output_dir
            
            # Throttled progress callback to prevent UI flooding
            def progress_callback(message: str, progress: float):
                nonlocal last_progress_update, processed_count
                
                # Check for cancellation
                if self._is_processing_cancelled():
                    return False  # Signal to stop
                    
                current_time = time.time()
                
                # Adaptive throttling based on batch size
                # For large batches, update less frequently
                if len(file_paths) > 1000:
                    update_interval = 2.0  # Update every 2 seconds for very large batches
                elif len(file_paths) > 500:
                    update_interval = 1.0  # Update every second for large batches
                elif len(file_paths) > 100:
                    update_interval = 0.5  # Update every 0.5s for medium batches
                else:
                    update_interval = 0.25  # Default for small batches
                
                # Only update progress at intervals or every N files
                update_every_n_files = max(10, len(file_paths) // 100)  # Update at most 100 times
                should_update = (current_time - last_progress_update >= update_interval) or \
                               (processed_count % update_every_n_files == 0)
                
                if should_update:
                    last_progress_update = current_time
                    if self.progress_dialog:
                        self._safe_after(0, lambda m=message, p=progress: self.progress_dialog.update_progress(m, p))
                    
                    # Don't force GUI updates for every file - let the main loop handle it
                    if processed_count % 50 == 0:  # Only force update every 50 files
                        self._safe_after(0, self.update)
                    
                    # Yield CPU time less frequently for large batches
                    if processed_count % 100 == 0:
                        time.sleep(0.001)
                    
                return True  # Continue processing
            
            # Enhanced progress callback with memory monitoring
            def enhanced_progress_callback(message: str, progress: float):
                nonlocal processed_count, last_gc_time
                
                # Check for cancellation first
                if self._is_processing_cancelled():
                    return False
                    
                current_time = time.time()
                
                # Update processed count based on progress
                processed_count = int(progress * len(file_paths))
                
                # Standard progress update with thread safety
                try:
                    if not progress_callback(message, progress):
                        return False
                except Exception as e:
                    logger.debug(f"Progress callback error (continuing): {e}")
                    # Continue processing even if UI update fails
                    pass
                
                # Adaptive memory management based on batch size
                # Clean up more frequently for large batches
                if len(file_paths) > 1000:
                    cleanup_interval = 25  # Every 25 files for very large batches
                    gc_time_interval = 20  # Every 20 seconds
                elif len(file_paths) > 500:
                    cleanup_interval = 35  # Every 35 files for large batches
                    gc_time_interval = 25  # Every 25 seconds
                else:
                    cleanup_interval = 50  # Default
                    gc_time_interval = 30  # Default
                
                # Memory management at adaptive intervals
                if (processed_count % cleanup_interval == 0 and processed_count > 0) or \
                   (current_time - last_gc_time > gc_time_interval):
                    logger.debug(f"Performing memory cleanup at file {processed_count}")
                    
                    # Force garbage collection
                    gc.collect()
                    last_gc_time = current_time
                    
                    # Clear any intermediate results from memory
                    import matplotlib.pyplot as plt
                    plt.close('all')  # Close all matplotlib figures
                    
                    # Clear processor cache if it's getting too large
                    if hasattr(self.processor, '_file_cache') and len(self.processor._file_cache) > 50:
                        # Keep only the most recent 25 entries
                        cache_keys = list(self.processor._file_cache.keys())
                        for key in cache_keys[:-25]:
                            del self.processor._file_cache[key]
                    
                    # Only yield CPU time during cleanup for very large batches
                    if len(file_paths) > 500:
                        time.sleep(0.005)  # Shorter sleep
                    
                return True
            
            # Initialize batch logger
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = output_dir / "batch_logs" if output_dir else Path.home() / ".laser_trim_analyzer" / "batch_logs"
            self.batch_logger = setup_batch_logging(batch_id, log_dir, enable_performance=True)
            
            # Log batch start
            batch_config = {
                'generate_plots': self.generate_plots_var.get(),
                'save_to_db': self.save_to_db_var.get(),
                'output_directory': str(output_dir) if output_dir else 'None',
                'file_count': len(file_paths),
                'concurrent_workers': 4
            }
            self.batch_logger.log_batch_start(len(file_paths), batch_config)
            
            # Run batch processing with optimizations
            try:
                
                # Limit max workers based on system resources and file count
                base_workers = 4  # Default workers since we removed the slider
                file_count = len(file_paths)
                
                # Scale down workers for very large batches to prevent resource exhaustion
                # IMPORTANT: Further reduced worker counts to prevent 100% CPU usage
                # Now always using single worker to prevent CPU overload
                max_workers = 1  # Always single worker to prevent 100% CPU
                
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
                        # Ask user if they want to keep plots enabled despite resource concerns
                        response = messagebox.askyesno(
                            "Resource Optimization",
                            "Plot generation uses significant memory for large batches.\n\n"
                            "Recommended: Disable plots to conserve memory.\n\n"
                            "Do you want to disable plot generation for this batch?"
                        )
                        if response:  # User agreed to disable
                            self.generate_plots_var.set(False)
                        # Otherwise keep plots enabled per user preference
                    
                    # Show resource warnings if any
                    if optimized_params.get('resource_warnings'):
                        warnings_text = "\n".join(optimized_params['resource_warnings'])
                        self._safe_after(0, lambda: messagebox.showwarning(
                            "Resource Warnings",
                            f"Resource constraints detected:\n\n{warnings_text}\n\n"
                            "Processing will continue with optimized settings."
                        ))
                
                logger.info(f"Processing {file_count} files with {max_workers} workers (optimized)")
                if self.batch_logger:
                    self.batch_logger.main_logger.info(f"Processing {file_count} files with {max_workers} workers (optimized)")
                
                # Get disable plots threshold from config
                disable_plots_threshold = getattr(self.analyzer_config.processing, 'disable_plots_threshold', 200)
                
                # Disable plots for very large batches to save memory
                if file_count > disable_plots_threshold and self.generate_plots_var.get():
                    # Create a thread-safe variable to store response
                    user_response = {'value': None}
                    
                    def ask_user():
                        response = messagebox.askyesno(
                            "Large Batch Detected",
                            f"Processing {file_count} files with plots enabled may cause performance issues.\n\n"
                            f"Plots are automatically disabled for batches over {disable_plots_threshold} files.\n\n"
                            "Do you want to disable plots for better performance?"
                        )
                        user_response['value'] = response
                    
                    # Ask user on main thread
                    self._safe_after(0, ask_user)
                    
                    # Wait for response (with timeout)
                    timeout = 10  # seconds
                    start_time = time.time()
                    while user_response['value'] is None and time.time() - start_time < timeout:
                        time.sleep(0.1)
                    
                    # If user agreed or timeout, disable plots
                    if user_response['value'] is None or user_response['value']:
                        self.generate_plots_var.set(False)
                        logger.info(f"Disabled plot generation for large batch ({file_count} files)")
                
                # Check if we should use turbo mode for batches >= 100 files
                # Enhanced debug logging for deployment troubleshooting
                logger.info(f"🔧 DEBUG: Starting turbo mode check for {file_count} files")
                logger.info(f"🔧 DEBUG: analyzer_config exists: {hasattr(self, 'analyzer_config')}")
                if hasattr(self, 'analyzer_config'):
                    logger.info(f"🔧 DEBUG: processing config exists: {hasattr(self.analyzer_config, 'processing')}")
                    if hasattr(self.analyzer_config, 'processing'):
                        logger.info(f"🔧 DEBUG: processing config type: {type(self.analyzer_config.processing)}")
                        logger.info(f"🔧 DEBUG: processing config attributes: {dir(self.analyzer_config.processing)}")
                
                turbo_threshold = getattr(self.analyzer_config.processing, 'turbo_mode_threshold', 100)
                logger.info(f"🔧 DEBUG: Retrieved turbo_threshold = {turbo_threshold}")
                logger.info(f"🔧 DEBUG: Comparison: {file_count} >= {turbo_threshold} = {file_count >= turbo_threshold}")
                
                if file_count >= turbo_threshold:
                    logger.info(f"✅ TURBO MODE ACTIVATED for {file_count} files (threshold: {turbo_threshold})")
                    logger.info(f"🔧 DEBUG: About to call _process_with_turbo_mode")
                    
                    # Disable plots for turbo mode
                    if self.generate_plots_var.get():
                        self.generate_plots_var.set(False)
                        logger.info("Plots automatically disabled for turbo mode")
                    
                    # Inform the user ML predictions are disabled in turbo mode
                    try:
                        self._safe_after(0, lambda: messagebox.showinfo(
                            "Turbo Mode",
                            "Turbo mode enabled for large batch.\n\n"
                            "For performance, ML predictions are disabled and heuristic risk estimates are used."
                        ))
                    except Exception:
                        pass
                    
                    # Use FastProcessor with turbo mode directly
                    results = self._process_with_turbo_mode(
                        file_paths=file_paths,
                        output_dir=output_dir,
                        progress_callback=enhanced_progress_callback
                    )
                else:
                    logger.info(f"❌ STANDARD MODE: {file_count} files below turbo threshold of {turbo_threshold}")
                    logger.info(f"🔧 DEBUG: Using standard processing with memory management")
                    # Standard processing with memory management
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
                self._safe_after(0, lambda: self._handle_batch_cancelled(results))
                return
            
            # Save to database if requested
            if self.save_to_db_var.get() and self._db_manager:
                self._save_batch_to_database(results)
            
            # Final cleanup
            gc.collect()
            
            # Check for cancellation one final time
            if self._is_processing_cancelled():
                self._safe_after(0, lambda: self._handle_batch_cancelled(results))
            else:
                # Update UI on main thread
                self._safe_after(0, lambda: self._handle_batch_success(results, output_dir))
                
        except ValidationError as e:
            logger.error(f"Batch validation error: {e}")
            error_msg = f"Batch validation failed:\n\n{str(e)}\n\nPlease check that all selected files are valid Excel files."
            self._safe_after(0, lambda: self._handle_batch_error(error_msg))
            
        except ProcessingError as e:
            logger.error(f"Batch processing error: {e}")
            # Error message is already formatted by the ProcessingError handler above
            self._safe_after(0, lambda: self._handle_batch_error(str(e)))
            
        except Exception as e:
            logger.error(f"Unexpected batch error: {e}")
            logger.error(traceback.format_exc())
            error_msg = f"An unexpected error occurred during batch processing:\n\n{str(e)}\n\n"
            error_msg += "Please check the log files for more details."
            if "MemoryError" in str(type(e).__name__):
                error_msg = f"Out of memory error:\n\n{str(e)}\n\nTry processing fewer files at once or disable plot generation."
            self._safe_after(0, lambda: self._handle_batch_error(error_msg))
        
        finally:
            with self._state_lock:
                self.is_processing = False
            
            # Finalize batch logging if available
            if self.batch_logger:
                try:
                    summary = self.batch_logger.finalize_batch()
                    logger.info(f"Batch processing summary saved to: {self.batch_logger.log_dir}")
                except Exception as e:
                    logger.error(f"Failed to finalize batch logger: {e}")
            
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
        self.failed_files = []  # Reset failed files tracking
        processed_files = 0
        total_files = len(file_paths)
        
        # Process in chunks to manage memory better
        if self.resource_manager:
            # Use adaptive chunk size based on resources
            chunk_size = self.resource_manager.get_adaptive_batch_size(total_files, 0)
        else:
            # Fallback to intelligent adaptive sizing based on batch size
            if total_files > 2000:
                chunk_size = 100  # Larger chunks for very large batches
            elif total_files > 1000:
                chunk_size = 75   # Medium-large chunks
            elif total_files > 500:
                chunk_size = 50   # Standard chunks for large batches
            elif total_files > 100:
                chunk_size = 25   # Smaller chunks for medium batches
            else:
                chunk_size = 10   # Small chunks for small batches
        
        # Ensure chunk_size is never 0 to prevent infinite loop
        chunk_size = max(1, chunk_size)
        
        # Log chunk strategy
        logger.info(f"Processing {total_files} files in chunks of {chunk_size}")
        
        for chunk_start in range(0, total_files, chunk_size):
            # Check for cancellation at start of each chunk
            if self._is_processing_cancelled():
                logger.info(f"Processing cancelled after {processed_files}/{total_files} files")
                # Important: Return accumulated results even on cancellation
                return results
                
            chunk_end = min(chunk_start + chunk_size, total_files)
            chunk_files = file_paths[chunk_start:chunk_end]
            
            logger.debug(f"Processing chunk {chunk_start}-{chunk_end} ({len(chunk_files)} files)")
            
            # Process chunk with ThreadPoolExecutor
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            try:
                # Submit files with controlled concurrency to prevent CPU overload
                future_to_file = {}
                active_futures = set()
                file_index = 0
                
                # Process files with limited concurrent submissions
                while file_index < len(chunk_files) or active_futures:
                    # Check for cancellation
                    if self._is_processing_cancelled():
                        break
                    
                    # Submit new files if we have capacity
                    while len(active_futures) < max_workers and file_index < len(chunk_files):
                        file_path = chunk_files[file_index]
                        future = executor.submit(self._process_single_file_safe, file_path, output_dir)
                        future_to_file[future] = file_path
                        active_futures.add(future)
                        file_index += 1
                        
                        # Increased delay between submissions to prevent CPU spike
                        time.sleep(0.2)  # 200ms between file submissions to reduce CPU load
                        
                        # Check CPU every 5 files
                        if file_index % 5 == 0 and self.resource_manager:
                            status = self.resource_manager.get_current_status()
                            if status.cpu_percent > 70:
                                logger.debug(f"CPU at {status.cpu_percent}%, adding extra delay")
                                time.sleep(1.0)  # Extra 1 second delay when CPU is high
                
                    # Wait for completed futures
                    if active_futures:
                        done, pending = concurrent.futures.wait(
                            active_futures, 
                            timeout=0.5,  # Check every 500ms
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        
                        # Process completed futures
                        for future in done:
                            active_futures.remove(future)
                            
                            if self._is_processing_cancelled():
                                logger.info("Cancellation requested, stopping file processing")
                                # Cancel remaining futures
                                for remaining_future in future_to_file:
                                    if not remaining_future.done():
                                        remaining_future.cancel()
                                break
                            
                            file_path = future_to_file[future]
                        
                        # Check if this future was cancelled
                        if future.cancelled():
                            logger.debug(f"Skipping cancelled future for {file_path}")
                            continue
                        
                        processed_files += 1
                        
                        # Log file start if batch logger available
                        file_start_time = time.time() if self.batch_logger else None
                        
                        try:
                            result = future.result()
                            if result is not None:
                                results[str(file_path)] = result
                                # Log successful completion
                                if self.batch_logger and file_start_time:
                                    self.batch_logger.log_file_complete(file_path, file_start_time, result)
                                
                                # Update UI with partial results adaptively
                                # Update less frequently for large batches
                                if total_files > 1000:
                                    update_interval = 50  # Update every 50 files for very large batches
                                elif total_files > 500:
                                    update_interval = 25  # Update every 25 files for large batches
                                else:
                                    update_interval = 10  # Default for smaller batches
                                
                                if len(results) % update_interval == 0:
                                    # Update results widget on main thread
                                    partial_results = results.copy()
                                    self._safe_after(0, lambda r=partial_results: self.batch_results_widget.display_results(r))
                            else:
                                self.failed_files.append((str(file_path), "Processing returned None"))
                                # Log failure
                                if self.batch_logger and file_start_time:
                                    self.batch_logger.log_file_complete(
                                        file_path, file_start_time, 
                                        error=Exception("Processing returned None")
                                    )
                                
                        except Exception as e:
                            logger.error(f"File processing failed for {file_path}: {e}")
                            self.failed_files.append((str(file_path), str(e)))
                            # Log error to batch logger
                            if self.batch_logger:
                                self.batch_logger.log_file_complete(file_path, file_start_time or time.time(), error=e)
                            # Don't show individual errors during batch - they'll be summarized at the end
                        
                        # Update progress
                        progress_fraction = processed_files / total_files
                        # Ensure progress never exceeds 1.0 (100%)
                        progress_fraction = min(progress_fraction, 1.0)
                        progress_percent = progress_fraction * 100
                        message = f"Processing {file_path.name} ({processed_files}/{total_files})"
                        
                        # Log batch progress periodically
                        if self.batch_logger and processed_files % 10 == 0:
                            self.batch_logger.log_batch_progress()
                        
                        # Call progress callback with fraction (0-1) not percentage
                        if not progress_callback(message, progress_fraction):
                            logger.info("Progress callback signaled to stop processing")
                            break
                        
                        # Add small delay after each file to reduce CPU load
                        time.sleep(0.1)  # 100ms delay after each file completion
                        
                        # Additional CPU check after each file
                        if processed_files % 3 == 0 and self.resource_manager:
                            status = self.resource_manager.get_current_status()
                            if status.cpu_percent > 60:
                                logger.debug(f"CPU at {status.cpu_percent}% after file {processed_files}, pausing")
                                time.sleep(0.5)  # Extra 500ms when CPU is moderately high
                            
            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout processing files in chunk {chunk_start}-{chunk_end}")
                # Continue with next chunk
            except Exception as e:
                logger.error(f"Error collecting results: {e}")
                # Continue processing if possible
            finally:
                # Ensure executor is properly shut down
                executor.shutdown(wait=False)  # Don't wait if cancelled
                if self._is_processing_cancelled():
                    logger.info("Executor shut down due to cancellation")
            
            # Memory cleanup after each chunk
            if self.resource_manager:
                # Check if we should pause for resources
                if self.resource_manager.should_pause_processing():
                    logger.info("Pausing for resource recovery...")
                    
                    # Update UI to show pause
                    if progress_callback:
                        progress_fraction = processed_files / total_files
                        # Ensure progress never exceeds 1.0 (100%)
                        progress_fraction = min(progress_fraction, 1.0)
                        progress_callback(
                            "Pausing for memory recovery...",
                            progress_fraction
                        )
                    
                    # Wait for resources with cancellation check
                    # Use a shorter timeout and check for cancellation
                    wait_start = time.time()
                    max_wait = 5.0  # Maximum 5 seconds wait
                    
                    while time.time() - wait_start < max_wait:
                        # Check if cancelled
                        if self._is_processing_cancelled():
                            logger.info("Cancellation detected during resource wait")
                            break
                            
                        # Check resources
                        status = self.resource_manager.get_current_status()
                        if not status.memory_critical and status.available_memory_mb > self.resource_manager.MIN_FREE_MEMORY_MB:
                            logger.info("Resources available, resuming processing")
                            break
                            
                        # Force cleanup
                        self.resource_manager.force_cleanup()
                        
                        # Short sleep to avoid blocking
                        time.sleep(0.1)
                    
                    if time.time() - wait_start >= max_wait:
                        logger.warning("Resource recovery timeout, continuing anyway")
                
                # Adaptive chunk size for next iteration
                if chunk_end < total_files:
                    next_chunk_size = self.resource_manager.get_adaptive_batch_size(
                        total_files, chunk_end
                    )
                    # Ensure chunk size is never 0
                    next_chunk_size = max(1, next_chunk_size)
                    logger.debug(f"Next chunk size: {next_chunk_size}")
                    # Note: We don't update chunk_size here as it's used in the range() iterator
            
            # Force cleanup
            import gc
            gc.collect()
            
            # Log memory cleanup if batch logger available
            if self.batch_logger:
                self.batch_logger.log_memory_cleanup(force_gc=True)
            
            # Significant delay between chunks to prevent CPU overload and system crashes
            # Much longer delays to give CPU time to cool down
            if chunk_size > 50:
                time.sleep(2.0)  # 2 seconds for large chunks
            elif chunk_size > 25:
                time.sleep(1.5)  # 1.5 seconds for medium chunks
            else:
                time.sleep(1.0)  # 1 second for small chunks
            
            # Additional CPU check - if resource manager indicates high CPU, wait longer
            if self.resource_manager:
                status = self.resource_manager.get_current_status()
                if status.cpu_high:
                    logger.info(f"High CPU detected ({status.cpu_percent}%), pausing for recovery")
                    time.sleep(5.0)  # 5 second pause when CPU is high
                elif status.cpu_percent > 50:
                    # Even if not "high", throttle for moderate CPU usage
                    logger.info(f"Moderate CPU usage ({status.cpu_percent}%), adding delay")
                    time.sleep(3.0)  # 3 second pause for moderate CPU
        
        if self.failed_files:
            logger.warning(f"Processing completed with {len(self.failed_files)} failures")
            # Show summary of all failures at the end
            if len(self.failed_files) > 0:
                failure_summary = "The following files failed to process:\n\n"
                for file_path, error in self.failed_files[:10]:  # Show first 10 errors
                    file_name = Path(file_path).name
                    failure_summary += f"• {file_name}: {error}\n"
                
                if len(self.failed_files) > 10:
                    failure_summary += f"\n... and {len(self.failed_files) - 10} more files"
                
                self._safe_after(0, lambda: messagebox.showwarning(
                    "Processing Failures",
                    failure_summary + "\n\nCheck the log files for complete details."
                ))
            
        return results
    
    def _process_with_turbo_mode(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path],
        progress_callback: Callable[[str, float], None]
    ) -> Dict[str, AnalysisResult]:
        """Process files using turbo mode with FastProcessor directly."""
        logger.info(f"🚀 TURBO MODE: Initializing processing for {len(file_paths)} files")
        logger.info(f"🔧 DEBUG: output_dir = {output_dir}")
        logger.info(f"🔧 DEBUG: About to import FastProcessor")
        
        # Import FastProcessor directly
        try:
            from laser_trim_analyzer.core.fast_processor import FastProcessor
            logger.info(f"🔧 DEBUG: FastProcessor imported successfully")
        except ImportError as e:
            logger.error(f"🚨 ERROR: Failed to import FastProcessor: {e}")
            raise
        
        # Create fast processor with turbo mode enabled
        logger.info(f"🔧 DEBUG: Creating FastProcessor with turbo_mode=True")
        try:
            fast_processor = FastProcessor(self.analyzer_config, turbo_mode=True)
            logger.info(f"🔧 DEBUG: FastProcessor created successfully")
            logger.info(f"🔧 DEBUG: FastProcessor turbo_mode: {fast_processor.turbo_mode}")
            logger.info(f"🔧 DEBUG: FastProcessor max_workers: {fast_processor.max_workers}")
        except Exception as e:
            logger.error(f"🚨 ERROR: Failed to create FastProcessor: {e}")
            raise
        
        # Track stats for UI updates
        self.stats = {
            'total_files': len(file_paths),
            'processed_files': 0,
            'failed_files': 0,
            'start_time': time.time()
        }
        
        # Create progress callback wrapper that checks for cancellation
        def turbo_progress_callback(message: str, progress: float):
            """Progress callback for turbo mode."""
            # Check for cancellation
            if self._is_processing_cancelled():
                logger.info("Turbo mode processing cancelled by user")
                return False  # Signal to stop processing
            
            # Update stats based on progress
            self.stats['processed_files'] = int(progress * len(file_paths))
            
            # Schedule UI update on main thread to prevent "main thread is not in main loop" error
            def update_ui():
                progress_callback(message, progress)
            
            try:
                self._safe_after(0, update_ui)
            except Exception as e:
                logger.debug(f"Could not schedule UI update: {e}")
                # Continue processing even if UI update fails
            
            return True  # Continue processing
        
        try:
            # Process files directly with FastProcessor (no async needed)
            results_list = fast_processor.process_batch_fast(
                file_paths,
                output_dir,
                turbo_progress_callback
            )
            
            logger.info(f"TURBO MODE completed: {len(results_list)} files processed")
            
            # Convert list to dict keyed by file path
            results = {}
            for result in results_list:
                if result and hasattr(result, 'metadata') and hasattr(result.metadata, 'file_path'):
                    results[str(result.metadata.file_path)] = result
            
            # Save to database if enabled
            if self.save_to_db_var.get() and self._db_manager and results:
                try:
                    # Save in batches
                    batch_size = 100
                    all_results = list(results.values())
                    for i in range(0, len(all_results), batch_size):
                        batch = all_results[i:i + batch_size]
                        if hasattr(self._db_manager, 'save_analysis_batch'):
                            force_overwrite = self.force_reprocess_var.get()
                            self._db_manager.save_analysis_batch(batch, force_overwrite=force_overwrite)
                        else:
                            # Fallback to individual saves
                            for result in batch:
                                self._db_manager.save_analysis_result(result)
                    logger.info(f"Saved {len(results)} results to database")
                except Exception as e:
                    logger.error(f"Failed to save turbo mode results to database: {e}")
            
            return results
                
        except Exception as e:
            logger.error(f"Turbo mode processing failed: {e}")
            raise ProcessingError(f"Turbo mode failed: {e}")
    
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
                self._safe_after(0, lambda: messagebox.showerror("File Processing Error", error_msg))
            elif self._file_error_count == 4:
                self._safe_after(0, lambda: messagebox.showwarning(
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
            self.export_excel_button.configure(state="normal")
            self.export_html_button.configure(state="normal")
            self.export_csv_button.configure(state="normal")
            
            # Enable output folder button if we have an output directory
            if hasattr(self, 'last_output_dir') and self.last_output_dir:
                self.output_folder_button.configure(state="normal")
        
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
            self._safe_after(0, lambda: messagebox.showerror(
                "Database Error",
                "Database is not initialized. Results could not be saved.\n\n"
                "Please check the database configuration and restart the application."
            ))
            return
            
        saved_count = 0
        failed_count = 0
        duplicate_count = 0
        
        logger.info(f"Starting database save for {len(results)} results")
        
        # Convert results to list
        result_list = list(results.values())
        file_paths = list(results.keys())
        
        # Pre-filter duplicates within the batch itself
        logger.info(f"Checking for duplicates within batch of {len(result_list)} files...")
        seen_combinations = {}  # Maps (model, serial, date) to (result_index, db_id)
        unique_indices = []  # Indices of unique results
        within_batch_duplicates = 0
        
        for i, result in enumerate(result_list):
            if result is None or not result.metadata:
                # Track these as failed files
                if i < len(file_paths):
                    file_name = Path(file_paths[i]).name
                    error_msg = "Invalid result: missing metadata" if result else "Result is None"
                    self.failed_files.append((file_name, error_msg))
                    logger.warning(f"Skipping invalid result for {file_name}: {error_msg}")
                else:
                    logger.warning(f"Skipping invalid result at index {i}: missing metadata")
                continue
                
            # Create unique key for this result
            key = (result.metadata.model, result.metadata.serial, result.metadata.file_date)
            
            if key in seen_combinations:
                within_batch_duplicates += 1
                first_idx, first_db_id = seen_combinations[key]
                logger.info(f"Duplicate within batch: {Path(file_paths[i]).name} is duplicate of "
                          f"{Path(file_paths[first_idx]).name} ({key[0]}-{key[1]})")
                # Will update this result's db_id after the first one is saved
            else:
                seen_combinations[key] = (i, None)  # Store index, db_id will be set later
                unique_indices.append(i)
        
        if within_batch_duplicates > 0:
            logger.warning(f"Found {within_batch_duplicates} duplicates within the batch itself!")
            logger.info(f"Will process {len(unique_indices)} unique results from {len(result_list)} total files")
        
        # Use batch save if available for better performance
        if hasattr(self._db_manager, 'save_analysis_batch'):
            try:
                # Check for cancellation
                if self._is_processing_cancelled():
                    logger.info("Cancellation requested during database save")
                    return
                
                # Process only unique results
                unique_results = [result_list[i] for i in unique_indices]
                
                if unique_results:
                    # Check Force Reprocess setting
                    force_reprocess = self.force_reprocess_var.get()
                    
                    if force_reprocess:
                        # Skip duplicate checking - process all files
                        results_to_save = [(unique_indices[i], result) for i, result in enumerate(unique_results)]
                        logger.info(f"Force reprocess enabled - processing all {len(results_to_save)} files")
                        
                        # Update progress to show we're skipping duplicate checking
                        if self.progress_dialog:
                            self._safe_after(0, lambda: self.progress_dialog.update_progress(
                                "Force reprocess - skipping duplicate check", 1.0))
                    else:
                        # Check for existing duplicates in database with progress tracking
                        results_to_save = []
                        db_duplicate_indices = []
                        total_to_check = len(unique_results)
                        
                        for check_idx, (idx, result) in enumerate(zip(unique_indices, unique_results)):
                            # Update progress during duplicate checking
                            if self.progress_dialog:
                                progress = check_idx / total_to_check if total_to_check > 0 else 0
                                status_msg = f"Checking for duplicates: {check_idx + 1}/{total_to_check}"
                                self._safe_after(0, lambda m=status_msg, p=progress: 
                                         self.progress_dialog.update_progress(m, p))
                            
                            existing_id = self._db_manager.check_duplicate_analysis(
                                result.metadata.model,
                                result.metadata.serial,
                                result.metadata.file_date
                            )
                            
                            if existing_id:
                                duplicate_count += 1
                                result.db_id = existing_id
                                # Update seen_combinations with the existing ID
                                key = (result.metadata.model, result.metadata.serial, result.metadata.file_date)
                                seen_combinations[key] = (idx, existing_id)
                                db_duplicate_indices.append(idx)
                                logger.info(f"Database duplicate: {Path(file_paths[idx]).name} already exists "
                                          f"with ID {existing_id} ({key[0]}-{key[1]})")
                            else:
                                results_to_save.append((idx, result))
                    
                    if duplicate_count > 0:
                        logger.info(f"Found {duplicate_count} existing analyses in database")
                    
                    # Save new results
                    if results_to_save:
                        logger.info(f"Attempting batch save of {len(results_to_save)} new results...")
                        results_only = [r[1] for r in results_to_save]
                        force_overwrite = self.force_reprocess_var.get()
                        analysis_ids = self._db_manager.save_analysis_batch(results_only, force_overwrite=force_overwrite)
                        
                        # Update saved results with database IDs
                        for (idx, result), db_id in zip(results_to_save, analysis_ids):
                            if db_id is not None:
                                result.db_id = db_id
                                saved_count += 1
                                # Update seen_combinations
                                key = (result.metadata.model, result.metadata.serial, result.metadata.file_date)
                                seen_combinations[key] = (idx, db_id)
                            else:
                                failed_count += 1
                        
                        logger.info(f"Batch saved {saved_count} new analyses to database")
                    
                    # Update all results (including within-batch duplicates) with their database IDs
                    for i, result in enumerate(result_list):
                        if result and result.metadata:
                            key = (result.metadata.model, result.metadata.serial, result.metadata.file_date)
                            if key in seen_combinations:
                                _, db_id = seen_combinations[key]
                                if db_id is not None and not hasattr(result, 'db_id'):
                                    result.db_id = db_id
            except Exception as e:
                logger.error(f"Batch database save failed: {e}")
                # Fall back to individual saves
                logger.info("Falling back to individual saves")
                self._safe_after(0, lambda: messagebox.showwarning(
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

        # Mark successfully processed files for incremental processing (Phase 1 Day 2)
        if self._db_manager and hasattr(self._db_manager, 'mark_file_processed'):
            marked_count = 0
            for file_path, result in results.items():
                try:
                    if result and result.db_id:
                        self._db_manager.mark_file_processed(
                            file_path=file_path,
                            success=True,
                            analysis_id=result.db_id
                        )
                        marked_count += 1
                except Exception as e:
                    # Non-critical - log and continue
                    logger.debug(f"Could not mark file as processed: {e}")
            if marked_count > 0:
                logger.info(f"Marked {marked_count} files as processed for incremental mode")

        if failed_count > 0:
            # Show warning about failed saves
            self._safe_after(0, lambda: messagebox.showwarning(
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
            # Save analysis to database
            result.db_id = self._db_manager.save_analysis(result)
            logger.info(f"Saved {Path(file_path).name} to database with ID: {result.db_id}")

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
        failed_count = len(self.failed_files)  # Use actual failed files count
        
        # Log detailed statistics for debugging
        logger.info(f"Batch processing statistics:")
        logger.info(f"  Total selected files: {len(self.selected_files)}")
        logger.info(f"  Results received: {len(results)}")
        logger.info(f"  Failed files: {failed_count}")
        logger.info(f"  Success rate: {successful_count}/{len(self.selected_files)}")
        
        # Log the failed files for debugging
        if failed_count > 0:
            logger.info(f"Failed files ({failed_count} total):")
            for file_name, error in self.failed_files[:10]:  # Show first 10
                logger.info(f"  - {file_name}: {error}")
            if failed_count > 10:
                logger.info(f"  ... and {failed_count - 10} more")
        
        # Validation statistics
        validated_count = 0
        warning_count = 0
        failed_validation_count = 0
        

        
        for r in results.values():
            if hasattr(r, 'overall_validation_status'):
                val_status = r.overall_validation_status
                # Handle both enum and string values - check multiple possible formats
                if hasattr(val_status, 'value'):
                    val_str = val_status.value
                else:
                    val_str = str(val_status)
                
                # Normalize to uppercase for comparison
                val_str_upper = val_str.upper()
                
                if val_str_upper in ['VALIDATED', 'VALID']:
                    validated_count += 1
                elif val_str_upper in ['WARNING', 'WARN']:
                    warning_count += 1
                elif val_str_upper in ['FAILED', 'FAIL', 'NOT VALIDATED', 'NOT_VALIDATED']:
                    failed_validation_count += 1
        
        # Calculate processing time
        if hasattr(self, 'processing_start_time'):
            processing_time = time.time() - self.processing_start_time
            time_str = f"{processing_time:.1f}s" if processing_time < 60 else f"{processing_time/60:.1f}m"
        else:
            time_str = "N/A"
        
        # Calculate detailed statistics for master summary
        track_counts = []
        pass_counts = []
        fail_counts = []
        models_processed = set()
        serials_processed = set()
        
        for result in results.values():
            # Model and serial tracking
            if hasattr(result.metadata, 'model'):
                models_processed.add(result.metadata.model)
            if hasattr(result.metadata, 'serial'):
                serials_processed.add(result.metadata.serial)
            
            # Track counts
            if hasattr(result, 'tracks') and result.tracks:
                if isinstance(result.tracks, dict):
                    track_counts.append(len(result.tracks))
                    # Count pass/fail for each track
                    for track in result.tracks.values():
                        if hasattr(track, 'status'):
                            status = getattr(track.status, 'value', str(track.status))
                            if status in ['Pass', 'PASS']:
                                pass_counts.append(1)
                            else:
                                fail_counts.append(1)
                        elif hasattr(track, 'overall_status'):
                            status = getattr(track.status, 'value', str(track.status))
                            if status in ['Pass', 'PASS']:
                                pass_counts.append(1)
                            else:
                                fail_counts.append(1)
                else:
                    # Handle list format (from DB)
                    track_counts.append(len(result.tracks))
                    for track in result.tracks:
                        if hasattr(track, 'status'):
                            status = getattr(track.status, 'value', str(track.status))
                            if status in ['Pass', 'PASS']:
                                pass_counts.append(1)
                            else:
                                fail_counts.append(1)
                        elif hasattr(track, 'overall_status'):
                            status = getattr(track.status, 'value', str(track.status))
                            if status in ['Pass', 'PASS']:
                                pass_counts.append(1)
                            else:
                                fail_counts.append(1)
        
        # Calculate averages and totals
        total_tracks = sum(track_counts) if track_counts else 0
        avg_tracks_per_file = sum(track_counts) / len(track_counts) if track_counts else 0
        total_pass = sum(pass_counts) if pass_counts else 0
        total_fail = sum(fail_counts) if fail_counts else 0
        
        # Update master summary panel
        self._update_master_summary(
            total_files=len(self.selected_files),
            processed=successful_count,
            failed=failed_count,
            validated=validated_count,
            warnings=warning_count,
            validation_failed=failed_validation_count,
            processing_time=time_str,
            total_tracks=total_tracks,
            avg_tracks=avg_tracks_per_file,
            tracks_passed=total_pass,
            tracks_failed=total_fail,
            unique_models=len(models_processed),
            unique_serials=len(serials_processed),
            error_details=self.failed_files  # Pass error details
        )
        
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
        
        # Re-enable controls first (this won't affect export buttons anymore)
        self._set_controls_state("normal")
        
        # Enable export buttons
        self.export_excel_button.configure(state="normal")
        self.export_html_button.configure(state="normal")
        self.export_csv_button.configure(state="normal")
        
        # Enable output folder button and store the output directory
        if output_dir:
            self.last_output_dir = output_dir
            self.output_folder_button.configure(state="normal")
        
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
        
        # Emit batch complete event for home page with a delay to ensure DB operations are complete
        def emit_completion_event():
            if hasattr(self.main_window, 'emit_event'):
                event_data = {
                    'page': 'batch_processing',
                    'type': 'batch_complete',
                    'successful_count': successful_count,
                    'failed_count': failed_count,
                    'total_count': len(self.selected_files),
                    'status': 'Complete',
                    'results': results  # Include results for immediate display
                }
                self.main_window.emit_event('analysis_complete', event_data)
                logger.info("Emitted batch analysis_complete event")
            
            # Also refresh home page directly as backup
            try:
                if hasattr(self.main_window, 'pages') and 'home' in self.main_window.pages:
                    home_page = self.main_window.pages['home']
                    if hasattr(home_page, 'refresh'):
                        home_page.refresh()
                        logger.info("Refreshed home page after batch processing")
            except Exception as e:
                logger.debug(f"Failed to refresh home page: {e}")
        
        # Emit event after a short delay to ensure database operations are complete
        self._safe_after(500, emit_completion_event)
        
        logger.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed")
    
    def _update_master_summary(self, **kwargs):
        """Update the master summary panel with batch processing statistics."""
        # Check if we're shutting down to prevent widget access errors
        if self._shutting_down:
            return
        
        try:
            # Clear the summary frame safely
            for widget in self.summary_frame.winfo_children():
                if widget.winfo_exists():
                    widget.destroy()
        except tk.TclError:
            # Widget might already be destroyed, ignore
            logger.debug("Widget already destroyed during summary update")
            return
        
        # Check again before creating new widgets
        if self._shutting_down:
            return
            
        try:
            # Create a grid layout for the summary
            summary_container = ctk.CTkFrame(self.summary_frame, fg_color="transparent")
            summary_container.pack(fill='both', expand=True, padx=10, pady=10)
        except tk.TclError as e:
            logger.debug(f"Error creating summary container: {e}")
            return
        
        # Title
        title_label = ctk.CTkLabel(
            summary_container,
            text="Batch Processing Summary",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15), sticky='w')
        
        # Create summary items in a grid layout
        summary_items = [
            ("Processing Time:", kwargs.get('processing_time', 'N/A')),
            ("Total Files:", f"{kwargs.get('total_files', 0)}"),
            ("Successfully Processed:", f"{kwargs.get('processed', 0)}"),
            ("Failed to Process:", f"{kwargs.get('failed', 0)}"),
            ("", ""),  # Empty row for spacing
            ("Validation Summary:", ""),
            ("  • Validated:", f"{kwargs.get('validated', 0)}"),
            ("  • Warnings:", f"{kwargs.get('warnings', 0)}"),
            ("  • Failed Validation:", f"{kwargs.get('validation_failed', 0)}"),
            ("", ""),  # Empty row for spacing
            ("Track Analysis:", ""),
            ("  • Total Tracks:", f"{kwargs.get('total_tracks', 0)}"),
            ("  • Average per File:", f"{kwargs.get('avg_tracks', 0):.1f}"),
            ("  • Tracks Passed:", f"{kwargs.get('tracks_passed', 0)}"),
            ("  • Tracks Failed:", f"{kwargs.get('tracks_failed', 0)}"),
            ("", ""),  # Empty row for spacing
            ("Unique Models:", f"{kwargs.get('unique_models', 0)}"),
            ("Unique Serials:", f"{kwargs.get('unique_serials', 0)}")
        ]
        
        # Add summary items to grid
        row = 1
        for label_text, value_text in summary_items:
            if label_text == "" and value_text == "":
                # Empty row for spacing
                row += 1
                continue
            
            # Determine styling based on content
            if label_text.endswith(":") and not label_text.startswith("  "):
                # Main category labels
                label_font = ctk.CTkFont(size=12, weight="bold")
                value_font = ctk.CTkFont(size=12, weight="bold")
            elif label_text.startswith("  "):
                # Sub-items
                label_font = ctk.CTkFont(size=11)
                value_font = ctk.CTkFont(size=11)
            else:
                # Regular items
                label_font = ctk.CTkFont(size=12)
                value_font = ctk.CTkFont(size=12)
            
            # Check if we're shutting down before creating widgets
            if self._shutting_down:
                return
                
            try:
                # Create label
                label = ctk.CTkLabel(
                    summary_container,
                    text=label_text,
                    font=label_font,
                    anchor='w'
                )
                label.grid(row=row, column=0, sticky='w', padx=(0, 10), pady=2)
            except tk.TclError as e:
                logger.debug(f"Error creating summary label: {e}")
                return
            
            # Create value
            if value_text:
                # Determine color based on content
                text_color = None
                try:
                    # Try to extract numeric value from the text
                    numeric_value = int(value_text.split()[0]) if value_text else 0
                    
                    if "Failed" in label_text and numeric_value > 0:
                        text_color = "red"
                    elif "Warnings" in label_text and numeric_value > 0:
                        text_color = "orange"
                    elif "Validated" in label_text and "Failed" not in label_text:
                        text_color = "green"
                    elif "Passed" in label_text:
                        text_color = "green"
                except (ValueError, IndexError):
                    # If we can't parse a number, use default colors based on keywords
                    if "Failed" in label_text:
                        text_color = "red"
                    elif "Warning" in label_text:
                        text_color = "orange"
                    elif "Validated" in label_text or "Passed" in label_text:
                        text_color = "green"
                
                try:
                    value = ctk.CTkLabel(
                        summary_container,
                        text=value_text,
                        font=value_font,
                        text_color=text_color,
                        anchor='w'
                    )
                    value.grid(row=row, column=1, sticky='w', pady=2)
                except tk.TclError as e:
                    logger.debug(f"Error creating summary value: {e}")
                    return
            
            row += 1
        
        # Add a separator line
        if not self._shutting_down:
            try:
                separator = ctk.CTkFrame(summary_container, height=2)
                separator.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(15, 10))
                row += 1
            except tk.TclError as e:
                logger.debug(f"Error creating separator: {e}")
                return
        
        # Add error details section if there are failed files
        error_details = kwargs.get('error_details', [])
        if error_details:
            # Error section title
            if self._shutting_down:
                return
            try:
                error_title = ctk.CTkLabel(
                    summary_container,
                    text="Processing Errors:",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color="red",
                    anchor='w'
                )
                error_title.grid(row=row, column=0, columnspan=2, sticky='w', pady=(10, 5))
                row += 1
            except tk.TclError as e:
                logger.debug(f"Error creating error title: {e}")
                return
            
            # Create scrollable frame for errors if many
            if len(error_details) > 5:
                error_frame = ctk.CTkScrollableFrame(
                    summary_container,
                    height=100,
                    fg_color=("gray95", "gray15")
                )
                error_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=(10, 0), pady=(0, 10))
                error_parent = error_frame
            else:
                error_parent = summary_container
            
            # Display error details
            error_row = 0
            for file_path, error in error_details[:20]:  # Show max 20 errors
                file_name = Path(file_path).name
                error_text = f"• {file_name}: {error}"
                
                # Truncate long error messages
                if len(error_text) > 80:
                    error_text = error_text[:77] + "..."
                
                error_label = ctk.CTkLabel(
                    error_parent,
                    text=error_text,
                    font=ctk.CTkFont(size=10),
                    text_color="red",
                    anchor='w'
                )
                
                if error_parent == summary_container:
                    error_label.grid(row=row, column=0, columnspan=2, sticky='w', padx=(20, 0), pady=1)
                    row += 1
                else:
                    error_label.grid(row=error_row, column=0, sticky='w', padx=5, pady=1)
                    error_row += 1
            
            if len(error_details) > 20:
                more_label = ctk.CTkLabel(
                    error_parent if error_parent != summary_container else summary_container,
                    text=f"... and {len(error_details) - 20} more errors",
                    font=ctk.CTkFont(size=10, style="italic"),
                    text_color="gray",
                    anchor='w'
                )
                if error_parent == summary_container:
                    more_label.grid(row=row, column=0, columnspan=2, sticky='w', padx=(20, 0), pady=1)
                else:
                    more_label.grid(row=error_row, column=0, sticky='w', padx=5, pady=1)
        
        # Update the container to ensure proper sizing
        summary_container.update_idletasks()
    
    def _on_resource_update(self, status: Any):
        """Handle resource status updates."""
        if not self.resource_status_label:
            return
            
        # Update UI on main thread
        self._safe_after(0, lambda: self._update_resource_display(status))
    
    def _update_resource_display(self, status: Any):
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
        
        if not self._shutting_down:
            try:
                self.resource_status_label.configure(text=message, text_color=color)
            except tk.TclError:
                return  # Widget destroyed
        
        # Update optimization strategy
        if self.resource_optimizer and hasattr(self, 'selected_files'):
            strategy = self.resource_optimizer.get_processing_strategy(len(self.selected_files))
            if not self._shutting_down:
                try:
                    self.optimization_label.configure(
                        text=f"Optimization: {strategy['description']}"
                    )
                except tk.TclError:
                    pass  # Widget destroyed

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
            
            # Set stop flags (thread-safe)
            self._stop_event.set()
            with self._state_lock:
                self._processing_cancelled = True
            self.request_stop_processing()
            
            # Update UI immediately
            self.start_button.configure(text="Stopping...", state="disabled")
            self.stop_button.configure(state="disabled")
            
            # Update status
            self._update_batch_status("Stopping Processing...", "orange")
            
            # Force UI update to ensure buttons are disabled
            self.update_idletasks()
            
            # The processing thread will handle the actual stopping
            
    def _is_processing_cancelled(self) -> bool:
        """Check if processing has been cancelled (thread-safe)."""
        with self._state_lock:
            cancelled = self._processing_cancelled
        return cancelled or self._stop_event.is_set() or self.is_stop_requested()

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
            
            # Don't override the export buttons and output folder button states
            # They are managed separately based on whether results exist

    def _export_batch_results(self, format_type: str = 'excel'):
        """Export batch processing results in specified format."""
        if not self.batch_results:
            messagebox.showerror("Error", "No results to export")
            return
        
        # Log the number of results to help debug
        logger.info(f"Exporting {len(self.batch_results)} batch results as {format_type}")
        
        # Set file extension and types based on format
        if format_type == 'excel':
            default_ext = ".xlsx"
            file_types = [("Excel files", "*.xlsx"), ("All files", "*.*")]
        elif format_type == 'html':
            default_ext = ".html"
            file_types = [("HTML files", "*.html"), ("All files", "*.*")]
        elif format_type == 'csv':
            default_ext = ".csv"
            file_types = [("CSV files", "*.csv"), ("All files", "*.*")]
        else:
            messagebox.showerror("Error", f"Unsupported format: {format_type}")
            return
        
        # Ask for export location
        file_path = filedialog.asksaveasfilename(
            title=f"Export Batch Results as {format_type.upper()}",
            defaultextension=default_ext,
            filetypes=file_types
        )
        
        if file_path:
            try:
                path_obj = Path(file_path)
                
                if format_type == 'excel':
                    self._export_batch_excel(path_obj)
                elif format_type == 'html':
                    self._export_batch_html(path_obj)
                elif format_type == 'csv':
                    self._export_batch_csv(path_obj)
                
                messagebox.showinfo("Export Complete", f"Batch results exported to:\n{file_path}")
                logger.info(f"Batch results exported to: {file_path} (format: {format_type})")
                
                # Also generate a summary JSON report alongside
                if format_type in ['excel', 'html']:
                    summary_path = path_obj.with_suffix('.summary.json')
                    try:
                        results_list = list(self.batch_results.values())
                        self.report_generator.generate_summary_report(results_list, summary_path)
                        logger.info(f"Summary report generated: {summary_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate summary report: {e}")
                
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
        """Export batch results to Excel format using comprehensive report generator."""
        try:
            # Convert batch_results dict to list of AnalysisResult objects
            results_list = list(self.batch_results.values())
            
            # Validate we have actual results
            if not results_list:
                raise ValueError("No results found in batch_results")
            
            logger.info(f"Exporting {len(results_list)} results to Excel")
            
            # Get the include raw data option
            include_raw_data = self.include_raw_data_var.get()
            
            # Use the enhanced Excel exporter for comprehensive data export
            try:
                from laser_trim_analyzer.utils.enhanced_excel_export import EnhancedExcelExporter
                enhanced_exporter = EnhancedExcelExporter()
                
                # Use batch export method
                enhanced_exporter.export_batch_comprehensive(
                    results=results_list,
                    output_path=file_path,
                    include_individual_details=include_raw_data,
                    max_individual_sheets=10 if include_raw_data else 0
                )
                
                logger.info("Used enhanced Excel exporter for comprehensive data export")
            except ImportError:
                # Fallback to standard report generator
                logger.warning("Enhanced Excel exporter not available, using standard export")
                self.report_generator.generate_comprehensive_excel_report(
                    results=results_list,
                    output_path=file_path,
                    include_raw_data=include_raw_data
                )
            
            logger.info(f"Batch results exported using comprehensive report generator to: {file_path} (raw data: {include_raw_data})")
            
        except Exception as e:
            # If comprehensive report fails, fall back to legacy export method
            logger.warning(f"Comprehensive report generation failed, using legacy export: {e}")
            self._export_batch_excel_legacy(file_path)
    
    def _export_batch_html(self, file_path: Path):
        """Export batch results to HTML format."""
        try:
            # Convert batch_results dict to list of AnalysisResult objects
            results_list = list(self.batch_results.values())
            
            # Generate HTML report
            self.report_generator.generate_html_report(
                results=results_list,
                output_path=file_path,
                title=f"Batch Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            logger.info(f"Batch results exported to HTML: {file_path}")
            
        except Exception as e:
            logger.error(f"HTML export failed: {e}")
            raise
    
    def _export_batch_excel_legacy(self, file_path: Path):
        """Legacy export method for batch results to Excel format."""
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
                                track_status = getattr(track.status, 'value', str(track.status))
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
                    
                    # Individual track data - moved inside try block for safety
                    if hasattr(result, 'tracks') and result.tracks:
                        # Handle both dict and list formats
                        if isinstance(result.tracks, dict):
                            tracks_items = result.tracks.items()
                        else:
                            # For list format, create enumerated items
                            tracks_items = enumerate(result.tracks)
                        
                        for track_id, track in tracks_items:
                            try:
                                track_row = {
                                    'File': file_name,
                                    'Model': model,
                                    'Serial': serial,
                                    'Track_ID': str(track_id),
                                    'Track_Status': getattr(track.status, 'value', 'Unknown') if hasattr(track, 'status') else 'Unknown',
                                    'Sigma_Gradient': track.sigma_analysis.sigma_gradient if hasattr(track, 'sigma_analysis') and track.sigma_analysis else None,
                                    'Sigma_Threshold': track.sigma_analysis.sigma_threshold if hasattr(track, 'sigma_analysis') and track.sigma_analysis else None,
                                    'Sigma_Pass': track.sigma_analysis.sigma_pass if hasattr(track, 'sigma_analysis') and track.sigma_analysis else None,
                                    'Linearity_Spec': track.linearity_analysis.linearity_spec if hasattr(track, 'linearity_analysis') and track.linearity_analysis else None,
                                    'Linearity_Pass': track.linearity_analysis.linearity_pass if hasattr(track, 'linearity_analysis') and track.linearity_analysis else None,
                                    'Risk_Category': track.risk_category.value if hasattr(track, 'risk_category') else 'Unknown'
                                }
                                all_track_data.append(track_row)
                            except Exception as track_error:
                                logger.error(f"Error processing track {track_id} for {file_name}: {track_error}")
                                # Continue with next track
                                
                except Exception as e:
                    # Log error and continue with next result
                    logger.error(f"Error processing result for export: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Add minimal data for this file
                    summary_data.append({
                        'File': file_name,
                        'Model': 'Error',
                        'Serial': 'Error',
                        'Error': str(e)
                    })
                    # Show warning about specific file
                    self._safe_after(0, lambda fn=file_name, err=str(e): messagebox.showwarning(
                        "Export Data Warning",
                        f"Could not export complete data for {fn}:\n{err}\n\n"
                        "Partial data will be included in the export."
                    ))
            
            # Ensure we have at least some data to write
            if not summary_data:
                # Create a minimal summary if no data was processed successfully
                summary_data.append({
                    'File': 'No data available',
                    'Model': 'N/A',
                    'Serial': 'N/A',
                    'Status': 'Export failed - no processable data'
                })
            
            # Write summary sheet
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Batch Summary', index=False)
            
            # Write track details sheet only if we have data
            if all_track_data:
                tracks_df = pd.DataFrame(all_track_data)
                tracks_df.to_excel(writer, sheet_name='Track Details', index=False)
            
            # Calculate statistics with error handling
            total_tracks = 0
            tracks_passed = 0
            tracks_failed = 0
            files_validated = 0
            files_warning = 0
            files_failed_validation = 0
            
            for result in self.batch_results.values():
                try:
                    # Count tracks safely
                    if hasattr(result, 'tracks'):
                        if isinstance(result.tracks, dict):
                            total_tracks += len(result.tracks)
                            for track in result.tracks.values():
                                if hasattr(track, 'overall_status'):
                                    if getattr(track.status, 'value', '') == 'Pass':
                                        tracks_passed += 1
                                    else:
                                        tracks_failed += 1
                                elif hasattr(track, 'status'):
                                    if getattr(track.status, 'value', '') == 'Pass':
                                        tracks_passed += 1
                                    else:
                                        tracks_failed += 1
                        elif isinstance(result.tracks, list):
                            total_tracks += len(result.tracks)
                            for track in result.tracks:
                                if hasattr(track, 'overall_status'):
                                    if getattr(track.status, 'value', '') == 'Pass':
                                        tracks_passed += 1
                                    else:
                                        tracks_failed += 1
                                elif hasattr(track, 'status'):
                                    if getattr(track.status, 'value', '') == 'Pass':
                                        tracks_passed += 1
                                    else:
                                        tracks_failed += 1
                    
                    # Count validation status safely
                    if hasattr(result, 'overall_validation_status'):
                        val_status = getattr(result.overall_validation_status, 'value', '')
                        if val_status == 'VALIDATED':
                            files_validated += 1
                        elif val_status == 'WARNING':
                            files_warning += 1
                        elif val_status == 'FAILED':
                            files_failed_validation += 1
                            
                except Exception as stat_error:
                    logger.error(f"Error calculating statistics: {stat_error}")
            
            # Statistics sheet
            pass_rate = f"{(tracks_passed / total_tracks * 100):.1f}%" if total_tracks > 0 else "0%"
            success_rate = f"{(len(self.batch_results) / len(self.selected_files) * 100):.1f}%" if len(self.selected_files) > 0 else "0%"
            
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
                    success_rate,
                    total_tracks,
                    tracks_passed,
                    tracks_failed,
                    pass_rate,
                    files_validated,
                    files_warning,
                    files_failed_validation
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Ensure at least one sheet is visible by accessing the workbook
            # This is important to prevent the "at least one sheet must be visible" error
            if hasattr(writer, 'book'):
                writer.book.active = 0  # Make the first sheet (Batch Summary) active

    def _export_batch_csv(self, file_path: Path):
        """Export batch results to CSV format."""
        import pandas as pd
        
        # Create comprehensive CSV export
        export_data = []
        
        for file_path_str, result in self.batch_results.items():
            file_name = Path(file_path_str).name
            
            try:
                # Get metadata safely
                model = getattr(result.metadata, 'model', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'
                serial = getattr(result.metadata, 'serial', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'
                
                # Handle system_type safely
                system_type = 'Unknown'
                if hasattr(result, 'metadata') and hasattr(result.metadata, 'system_type'):
                    system_type = getattr(result.metadata.system_type, 'value', str(result.metadata.system_type))
                
                # Handle analysis_date safely
                analysis_date = 'Unknown'
                if hasattr(result, 'metadata') and hasattr(result.metadata, 'analysis_date'):
                    if hasattr(result.metadata.analysis_date, 'strftime'):
                        analysis_date = result.metadata.analysis_date.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        analysis_date = str(result.metadata.analysis_date)
                
                # Handle overall_status safely
                overall_status = getattr(result.overall_status, 'value', 'Unknown') if hasattr(result, 'overall_status') else 'Unknown'
                processing_time = f"{getattr(result, 'processing_time', 0):.2f}"
                validation_status = getattr(result.overall_validation_status, 'value', 'N/A') if hasattr(result, 'overall_validation_status') else 'N/A'
                
                # Process tracks
                if hasattr(result, 'tracks') and result.tracks:
                    # Handle both dict and list formats
                    if isinstance(result.tracks, dict):
                        tracks_items = result.tracks.items()
                    else:
                        tracks_items = enumerate(result.tracks)
                    
                    for track_id, track in tracks_items:
                        try:
                            row = {
                                'File': file_name,
                                'Model': model,
                                'Serial': serial,
                                'System_Type': system_type,
                                'Analysis_Date': analysis_date,
                                'Track_ID': str(track_id),
                                'Overall_Status': overall_status,
                                'Track_Status': getattr(track.overall_status, 'value', 'Unknown') if hasattr(track, 'overall_status') else getattr(track.status, 'value', 'Unknown') if hasattr(track, 'status') else 'Unknown',
                                'Processing_Time': processing_time,
                                'Validation_Status': validation_status
                            }
                            
                            # Add detailed analysis data
                            if hasattr(track, 'sigma_analysis') and track.sigma_analysis:
                                row.update({
                                    'Sigma_Gradient': getattr(track.sigma_analysis, 'sigma_gradient', None),
                                    'Sigma_Threshold': getattr(track.sigma_analysis, 'sigma_threshold', None),
                                    'Sigma_Pass': getattr(track.sigma_analysis, 'sigma_pass', None),
                                    'Sigma_Improvement': getattr(track.sigma_analysis, 'improvement_percent', None)
                                })
                            
                            if hasattr(track, 'linearity_analysis') and track.linearity_analysis:
                                row.update({
                                    'Linearity_Spec': getattr(track.linearity_analysis, 'linearity_spec', None),
                                    'Linearity_Pass': getattr(track.linearity_analysis, 'linearity_pass', None),
                                    'Linearity_Error': getattr(track.linearity_analysis, 'linearity_error', None)
                                })
                            
                            if hasattr(track, 'resistance_analysis') and track.resistance_analysis:
                                row.update({
                                    'Resistance_Before': getattr(track.resistance_analysis, 'resistance_before', None),
                                    'Resistance_After': getattr(track.resistance_analysis, 'resistance_after', None),
                                    'Resistance_Change_Percent': getattr(track.resistance_analysis, 'resistance_change_percent', None)
                                })
                            
                            if hasattr(track, 'risk_category'):
                                row['Risk_Category'] = getattr(track.risk_category, 'value', 'Unknown')
                            
                            export_data.append(row)
                            
                        except Exception as track_error:
                            logger.error(f"Error processing track {track_id} for CSV export: {track_error}")
                            # Add minimal row for this track
                            export_data.append({
                                'File': file_name,
                                'Model': model,
                                'Serial': serial,
                                'Track_ID': str(track_id),
                                'Error': str(track_error)
                            })
                else:
                    # No tracks - add a summary row
                    export_data.append({
                        'File': file_name,
                        'Model': model,
                        'Serial': serial,
                        'System_Type': system_type,
                        'Analysis_Date': analysis_date,
                        'Overall_Status': overall_status,
                        'Processing_Time': processing_time,
                        'Validation_Status': validation_status,
                        'Track_Count': 0
                    })
                    
            except Exception as e:
                logger.error(f"Error processing result for CSV export: {e}")
                # Add error row
                export_data.append({
                    'File': file_name,
                    'Error': str(e)
                })
        
        # Ensure we have at least one row
        if not export_data:
            export_data.append({
                'File': 'No data available',
                'Status': 'Export failed - no processable data'
            })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(file_path, index=False)

    def _clear_results(self):
        """Clear batch processing results."""
        self.batch_results = {}
        self.failed_files = []  # Clear failed files tracking
        self.batch_results_widget.clear()
        self.export_excel_button.configure(state="disabled")
        self.export_html_button.configure(state="disabled")
        self.export_csv_button.configure(state="disabled")
        self.output_folder_button.configure(state="disabled")
        self.last_output_dir = None
        
        # Clear summary panel
        if hasattr(self, 'summary_label'):
            self.summary_label.configure(text="Process files to see summary statistics")
        
        logger.info("Batch results cleared")
    
    def _open_output_folder(self):
        """Open the output folder containing batch results."""
        if hasattr(self, 'last_output_dir') and self.last_output_dir:
            try:
                # Platform-specific folder opening
                import platform
                import subprocess
                
                system = platform.system()
                if system == "Windows":
                    os.startfile(str(self.last_output_dir))
                elif system == "Darwin":  # macOS
                    subprocess.run(["open", str(self.last_output_dir)])
                else:  # Linux and others
                    subprocess.run(["xdg-open", str(self.last_output_dir)])
                    
                logger.info(f"Opened output folder: {self.last_output_dir}")
            except Exception as e:
                logger.error(f"Failed to open output folder: {e}")
                messagebox.showerror("Error", f"Failed to open output folder:\n{str(e)}")
        else:
            # Fallback to data directory
            try:
                data_dir = Path(self.analyzer_config.data_directory)
                batch_dir = data_dir / "batch_processing"
                
                if batch_dir.exists():
                    import platform
                    import subprocess
                    
                    system = platform.system()
                    if system == "Windows":
                        os.startfile(str(batch_dir))
                    elif system == "Darwin":  # macOS
                        subprocess.run(["open", str(batch_dir)])
                    else:  # Linux and others
                        subprocess.run(["xdg-open", str(batch_dir)])
                else:
                    messagebox.showinfo("Info", "No batch processing results found yet.")
            except Exception as e:
                logger.error(f"Failed to open batch folder: {e}")
                messagebox.showerror("Error", f"Failed to open batch folder:\n{str(e)}")
    
    def cleanup(self):
        """Cleanup resources when page is destroyed."""
        # Stop any ongoing processing
        if self.is_processing:
            self._stop_processing()
        
        # Cleanup resource manager callbacks
        if self.resource_manager:
            # Remove our callback (would need to track it properly in production)
            pass
            
    def show(self):
        """Show the page using grid layout."""
        self.grid(row=0, column=0, sticky="nsew")
        self.is_visible = True
        
        # Refresh if needed
        if self.needs_refresh:
            self.refresh()
            self.needs_refresh = False
            
        self.on_show()
        
    def hide(self):
        """Hide the page."""
        self.grid_remove()
        self.is_visible = False
        
    def refresh(self):
        """Refresh page content."""
        pass
        
    def on_show(self):
        """Called when page is shown."""
        pass
        
    def _create_error_page(self, error_msg: str):
        """Create an error display page."""
        error_frame = ctk.CTkFrame(self)
        error_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        error_label = ctk.CTkLabel(
            error_frame,
            text=f"Error initializing {self.__class__.__name__}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="red"
        )
        error_label.pack(pady=(20, 10))
        
        detail_label = ctk.CTkLabel(
            error_frame,
            text=error_msg,
            font=ctk.CTkFont(size=14),
            wraplength=600
        )
        detail_label.pack(pady=10)
    
    def _safe_after(self, delay, callback):
        """Safely schedule an after callback with error handling."""
        if self._shutting_down:
            return
        try:
            if self.winfo_exists():
                self.after(delay, callback)
        except Exception:
            pass  # Widget destroyed, ignore
    
    def cleanup(self):
        """Cleanup method to prevent callback errors during shutdown."""
        self._shutting_down = True
        self._stop_requested = True
        
        # Signal processing threads to stop
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        
        # Clear batch results to stop operations
        if hasattr(self, 'batch_results'):
            self.batch_results.clear()
            
        # Cleanup batch results widget
        if hasattr(self, 'batch_results_widget') and hasattr(self.batch_results_widget, 'cleanup'):
            try:
                self.batch_results_widget.cleanup()
            except Exception:
                pass 
