"""
Progress Dialog Widgets

Provides progress dialogs for single file and batch processing operations.
"""

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ProgressDialog:
    """Progress dialog for single file processing."""

    def __init__(self, parent, title: str = "Processing", message: str = "Please wait..."):
        """Initialize progress dialog."""
        self.parent = parent
        self.title = title
        self.message = message
        
        # Create dialog window
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        
        # Center on parent
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Make dialog modal
        self.dialog.focus_set()
        
        # Create widgets
        self._create_widgets()
        
        # State
        self.is_visible = False

    def _create_widgets(self):
        """Create progress dialog widgets."""
        # Main frame
        self.main_frame = ctk.CTkFrame(self.dialog)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Message label
        self.message_label = ctk.CTkLabel(
            self.main_frame,
            text=self.message,
            font=ctk.CTkFont(size=14)
        )
        self.message_label.pack(pady=(0, 20))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self.main_frame,
            width=300,
            height=20
        )
        self.progress_bar.pack(pady=(0, 10))
        self.progress_bar.set(0)
        
        # Progress percentage label
        self.progress_label = ctk.CTkLabel(
            self.main_frame,
            text="0%",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack()

    def show(self):
        """Show the progress dialog."""
        if not self.is_visible:
            self.dialog.deiconify()
            self.is_visible = True
            
            # Center on parent
            self._center_on_parent()

    def hide(self):
        """Hide the progress dialog."""
        if self.is_visible:
            self.dialog.withdraw()
            self.is_visible = False

    def destroy(self):
        """Destroy the progress dialog."""
        if hasattr(self, 'dialog'):
            self.dialog.destroy()

    def update_progress(self, message: str, progress: float):
        """Update progress dialog."""
        if self.is_visible:
            self.message_label.configure(text=message)
            self.progress_bar.set(progress)
            self.progress_label.configure(text=f"{progress * 100:.1f}%")
            
            # Force update
            self.dialog.update_idletasks()

    def _center_on_parent(self):
        """Center dialog on parent window."""
        try:
            # Get parent geometry
            parent_x = self.parent.winfo_rootx()
            parent_y = self.parent.winfo_rooty()
            parent_width = self.parent.winfo_width()
            parent_height = self.parent.winfo_height()
            
            # Calculate center position
            dialog_width = 400
            dialog_height = 150
            
            x = parent_x + (parent_width - dialog_width) // 2
            y = parent_y + (parent_height - dialog_height) // 2
            
            self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
            
        except Exception as e:
            logger.warning(f"Could not center progress dialog: {e}")


class BatchProgressDialog:
    """Progress dialog for batch processing operations."""

    def __init__(self, parent, title: str = "Batch Processing", total_files: int = 0):
        """Initialize batch progress dialog."""
        self.parent = parent
        self.title = title
        self.total_files = total_files
        self.current_file = 0
        
        # Create dialog window
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x200")
        self.dialog.resizable(False, False)
        
        # Center on parent
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Make dialog modal
        self.dialog.focus_set()
        
        # Create widgets
        self._create_widgets()
        
        # State
        self.is_visible = False

    def _create_widgets(self):
        """Create batch progress dialog widgets."""
        # Main frame
        self.main_frame = ctk.CTkFrame(self.dialog)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title label
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text=f"Processing {self.total_files} files...",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.pack(pady=(0, 20))
        
        # Current file label
        self.current_file_label = ctk.CTkLabel(
            self.main_frame,
            text="Initializing...",
            font=ctk.CTkFont(size=12)
        )
        self.current_file_label.pack(pady=(0, 10))
        
        # Overall progress bar
        self.overall_progress_label = ctk.CTkLabel(
            self.main_frame,
            text="Overall Progress:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.overall_progress_label.pack(anchor="w", pady=(10, 5))
        
        self.overall_progress_bar = ctk.CTkProgressBar(
            self.main_frame,
            width=400,
            height=20
        )
        self.overall_progress_bar.pack(pady=(0, 5))
        self.overall_progress_bar.set(0)
        
        # Overall progress percentage
        self.overall_progress_percent = ctk.CTkLabel(
            self.main_frame,
            text="0%",
            font=ctk.CTkFont(size=11)
        )
        self.overall_progress_percent.pack()
        
        # File progress bar
        self.file_progress_label = ctk.CTkLabel(
            self.main_frame,
            text="Current File Progress:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.file_progress_label.pack(anchor="w", pady=(10, 5))
        
        self.file_progress_bar = ctk.CTkProgressBar(
            self.main_frame,
            width=400,
            height=15
        )
        self.file_progress_bar.pack(pady=(0, 5))
        self.file_progress_bar.set(0)
        
        # File progress percentage
        self.file_progress_percent = ctk.CTkLabel(
            self.main_frame,
            text="0%",
            font=ctk.CTkFont(size=10)
        )
        self.file_progress_percent.pack()

    def show(self):
        """Show the batch progress dialog."""
        if not self.is_visible:
            self.dialog.deiconify()
            self.is_visible = True
            
            # Center on parent
            self._center_on_parent()

    def hide(self):
        """Hide the batch progress dialog."""
        if self.is_visible:
            self.dialog.withdraw()
            self.is_visible = False

    def destroy(self):
        """Destroy the batch progress dialog."""
        if hasattr(self, 'dialog'):
            self.dialog.destroy()

    def update_progress(self, message: str, overall_progress: float, file_progress: float = None):
        """Update batch progress dialog."""
        if self.is_visible:
            self.current_file_label.configure(text=message)
            self.overall_progress_bar.set(overall_progress)
            self.overall_progress_percent.configure(text=f"{overall_progress * 100:.1f}%")
            
            if file_progress is not None:
                self.file_progress_bar.set(file_progress)
                self.file_progress_percent.configure(text=f"{file_progress * 100:.1f}%")
            
            # Force update
            self.dialog.update_idletasks()

    def start_file(self, file_name: str, file_index: int):
        """Start processing a new file."""
        self.current_file = file_index
        self.current_file_label.configure(text=f"Processing: {file_name}")
        self.file_progress_bar.set(0)
        self.file_progress_percent.configure(text="0%")
        
        # Update overall progress based on file index
        if self.total_files > 0:
            overall_progress = file_index / self.total_files
            self.overall_progress_bar.set(overall_progress)
            self.overall_progress_percent.configure(text=f"{overall_progress * 100:.1f}%")

    def complete_file(self, file_name: str, file_index: int):
        """Complete processing of a file."""
        self.file_progress_bar.set(1.0)
        self.file_progress_percent.configure(text="100%")
        
        # Update overall progress
        if self.total_files > 0:
            overall_progress = (file_index + 1) / self.total_files
            self.overall_progress_bar.set(overall_progress)
            self.overall_progress_percent.configure(text=f"{overall_progress * 100:.1f}%")

    def _center_on_parent(self):
        """Center dialog on parent window."""
        try:
            # Get parent geometry
            parent_x = self.parent.winfo_rootx()
            parent_y = self.parent.winfo_rooty()
            parent_width = self.parent.winfo_width()
            parent_height = self.parent.winfo_height()
            
            # Calculate center position
            dialog_width = 500
            dialog_height = 200
            
            x = parent_x + (parent_width - dialog_width) // 2
            y = parent_y + (parent_height - dialog_height) // 2
            
            self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
            
        except Exception as e:
            logger.warning(f"Could not center batch progress dialog: {e}")


class SimpleProgressBar(ctk.CTkFrame):
    """Simple progress bar widget for embedding in other widgets."""

    def __init__(self, parent, **kwargs):
        """Initialize simple progress bar."""
        super().__init__(parent, **kwargs)
        
        # Create widgets
        self._create_widgets()
        
        # State
        self.progress = 0.0

    def _create_widgets(self):
        """Create progress bar widgets."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self,
            width=200,
            height=15
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.progress_bar.set(0)
        
        # Progress label
        self.progress_label = ctk.CTkLabel(
            self,
            text="0%",
            font=ctk.CTkFont(size=10)
        )
        self.progress_label.grid(row=1, column=0, pady=(0, 5))

    def set_progress(self, progress: float):
        """Set progress value (0.0 to 1.0)."""
        self.progress = max(0.0, min(1.0, progress))
        self.progress_bar.set(self.progress)
        self.progress_label.configure(text=f"{self.progress * 100:.1f}%")

    def reset(self):
        """Reset progress to zero."""
        self.set_progress(0.0)


class ProgressIndicator(ctk.CTkFrame):
    """Progress indicator with message and percentage."""

    def __init__(self, parent, title: str = "Progress", **kwargs):
        """Initialize progress indicator."""
        super().__init__(parent, **kwargs)
        
        self.title = title
        
        # Create widgets
        self._create_widgets()
        
        # State
        self.progress = 0.0

    def _create_widgets(self):
        """Create progress indicator widgets."""
        self.grid_rowconfigure([0, 1, 2], weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Title label
        self.title_label = ctk.CTkLabel(
            self,
            text=self.title,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.title_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        # Message label
        self.message_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=ctk.CTkFont(size=11)
        )
        self.message_label.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        # Progress frame
        self.progress_frame = ctk.CTkFrame(self)
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        self.progress_frame.grid_columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            width=150,
            height=12
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.progress_bar.set(0)
        
        # Progress percentage
        self.progress_percent = ctk.CTkLabel(
            self.progress_frame,
            text="0%",
            font=ctk.CTkFont(size=9)
        )
        self.progress_percent.grid(row=1, column=0, pady=(0, 5))

    def update(self, message: str, progress: float):
        """Update progress indicator."""
        self.message_label.configure(text=message)
        self.progress = max(0.0, min(1.0, progress))
        self.progress_bar.set(self.progress)
        self.progress_percent.configure(text=f"{self.progress * 100:.1f}%")

    def reset(self):
        """Reset progress indicator."""
        self.update("Ready", 0.0)

    def complete(self, message: str = "Complete"):
        """Mark as complete."""
        self.update(message, 1.0) 