"""
Progress Widgets - CustomTkinter Version

Progress dialogs and widgets for batch processing.
"""

import customtkinter as ctk
from typing import Optional
import logging


class BatchProgressDialog:
    """Progress dialog for batch processing operations."""
    
    def __init__(self, parent, title: str = "Processing", total_files: int = 0):
        self.parent = parent
        self.title = title
        self.total_files = total_files
        self.current_file = 0
        self.logger = logging.getLogger(__name__)
        
        self.dialog = None
        self.progress_bar = None
        self.status_label = None
        self.file_label = None
        
    def show(self):
        """Show the progress dialog."""
        try:
            self.dialog = ctk.CTkToplevel(self.parent)
            self.dialog.title(self.title)
            self.dialog.geometry("500x200")
            self.dialog.transient(self.parent)
            
            # Center the dialog
            self.dialog.update_idletasks()
            x = (self.dialog.winfo_screenwidth() - 500) // 2
            y = (self.dialog.winfo_screenheight() - 200) // 2
            self.dialog.geometry(f"500x200+{x}+{y}")
            
            # Prevent closing
            self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)
            
            # Main frame
            main_frame = ctk.CTkFrame(self.dialog)
            main_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            # Title
            title_label = ctk.CTkLabel(
                main_frame,
                text=self.title,
                font=ctk.CTkFont(size=16, weight="bold")
            )
            title_label.pack(pady=(0, 10))
            
            # File info
            self.file_label = ctk.CTkLabel(
                main_frame,
                text=f"Processing file 0 of {self.total_files}",
                font=ctk.CTkFont(size=12)
            )
            self.file_label.pack(pady=5)
            
            # Status
            self.status_label = ctk.CTkLabel(
                main_frame,
                text="Initializing...",
                font=ctk.CTkFont(size=11)
            )
            self.status_label.pack(pady=5)
            
            # Progress bar
            self.progress_bar = ctk.CTkProgressBar(main_frame, width=400)
            self.progress_bar.pack(pady=10)
            self.progress_bar.set(0)
            
            # Force update
            self.dialog.update()
            
        except Exception as e:
            self.logger.error(f"Error creating progress dialog: {e}")
            
    def update_progress(self, message: str, progress: float):
        """Update progress display."""
        try:
            if self.dialog and self.dialog.winfo_exists():
                # Update labels
                if self.status_label:
                    self.status_label.configure(text=message)
                    
                # Update file counter
                self.current_file = int(progress * self.total_files / 100)
                if self.file_label:
                    self.file_label.configure(
                        text=f"Processing file {self.current_file} of {self.total_files}"
                    )
                    
                # Update progress bar
                if self.progress_bar:
                    self.progress_bar.set(progress / 100.0)
                    
                # Force update
                self.dialog.update()
                
        except Exception as e:
            self.logger.debug(f"Error updating progress: {e}")
            
    def hide(self):
        """Hide and destroy the progress dialog."""
        try:
            if self.dialog and self.dialog.winfo_exists():
                self.dialog.destroy()
        except Exception as e:
            self.logger.debug(f"Error hiding progress dialog: {e}")
        finally:
            self.dialog = None