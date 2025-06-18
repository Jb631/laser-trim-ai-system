"""
Progress Widgets - CustomTkinter Version

Progress dialogs and widgets for batch processing.
"""

import customtkinter as ctk
from typing import Optional
import logging
import threading


class ProgressDialog:
    """Progress dialog for single file operations."""
    
    def __init__(self, parent, title: str = "Processing", message: str = "Processing..."):
        self.parent = parent
        self.title = title
        self.message = message
        self.logger = logging.getLogger(__name__)
        
        self.dialog = None
        self.progress_bar = None
        self.status_label = None
        
    def show(self):
        """Show the progress dialog."""
        try:
            self.dialog = ctk.CTkToplevel(self.parent)
            self.dialog.title(self.title)
            self.dialog.geometry("400x150")
            self.dialog.transient(self.parent)
            
            # Center the dialog
            self.dialog.update_idletasks()
            x = (self.dialog.winfo_screenwidth() - 400) // 2
            y = (self.dialog.winfo_screenheight() - 150) // 2
            self.dialog.geometry(f"400x150+{x}+{y}")
            
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
            
            # Status
            self.status_label = ctk.CTkLabel(
                main_frame,
                text=self.message,
                font=ctk.CTkFont(size=12)
            )
            self.status_label.pack(pady=5)
            
            # Progress bar
            self.progress_bar = ctk.CTkProgressBar(main_frame, width=300)
            self.progress_bar.pack(pady=10)
            self.progress_bar.set(0)
            self.progress_bar.start()  # Indeterminate progress
            
            # Force update
            self.dialog.update()
            
        except Exception as e:
            self.logger.error(f"Error creating progress dialog: {e}")
            
    def update_message(self, message: str):
        """Update the status message."""
        try:
            if self.dialog and self.dialog.winfo_exists():
                if self.status_label:
                    self.status_label.configure(text=message)
                self.dialog.update()
        except Exception as e:
            self.logger.debug(f"Error updating message: {e}")
    
    def update_progress(self, message: str, progress: float):
        """Update the progress bar and message (thread-safe)."""
        # Check if we're in the main thread
        if threading.current_thread() is not threading.main_thread():
            # Schedule update in main thread
            try:
                self.parent.after(0, lambda: self.update_progress(message, progress))
                return
            except Exception:
                # Parent might not have after method, continue anyway
                pass
        
        try:
            if self.dialog and self.dialog.winfo_exists():
                # Update message
                if self.status_label:
                    self.status_label.configure(text=message)
                
                # Update progress bar
                if self.progress_bar:
                    # Stop indeterminate mode if running
                    self.progress_bar.stop()
                    # Set determinate progress (0.0 to 1.0)
                    self.progress_bar.set(progress)
                
                # Force update
                self.dialog.update()
        except Exception as e:
            self.logger.debug(f"Error updating progress: {e}")
            
    def hide(self):
        """Hide and destroy the progress dialog."""
        try:
            if self.dialog and self.dialog.winfo_exists():
                if self.progress_bar:
                    self.progress_bar.stop()
                self.dialog.destroy()
        except Exception as e:
            self.logger.debug(f"Error hiding progress dialog: {e}")
        finally:
            self.dialog = None


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
                # Progress can be either 0-1 or 0-100, handle both
                if progress <= 1.0:
                    # Progress is 0-1
                    self.current_file = int(progress * self.total_files)
                    progress_normalized = progress
                else:
                    # Progress is 0-100
                    self.current_file = int(progress * self.total_files / 100)
                    progress_normalized = progress / 100.0
                    
                if self.file_label:
                    self.file_label.configure(
                        text=f"Processing file {self.current_file} of {self.total_files}"
                    )
                    
                # Update progress bar
                if self.progress_bar:
                    self.progress_bar.set(progress_normalized)
                    
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


class SimpleProgressBar(ctk.CTkFrame):
    """Simple progress bar widget."""
    
    def __init__(self, parent, width: int = 300, height: int = 20):
        super().__init__(parent)
        self.configure(fg_color="transparent")
        
        self.progress_bar = ctk.CTkProgressBar(self, width=width, height=height)
        self.progress_bar.pack()
        self.progress_bar.set(0)
        
    def set_progress(self, value: float):
        """Set progress value (0.0 to 1.0)."""
        self.progress_bar.set(value)
        
    def start(self):
        """Start indeterminate progress."""
        self.progress_bar.start()
        
    def stop(self):
        """Stop indeterminate progress."""
        self.progress_bar.stop()


class ProgressIndicator(ctk.CTkFrame):
    """Progress indicator with label."""
    
    def __init__(self, parent, text: str = "Processing..."):
        super().__init__(parent)
        self.configure(fg_color="transparent")
        
        self.label = ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=12))
        self.label.pack(pady=(0, 5))
        
        self.progress_bar = ctk.CTkProgressBar(self, width=200)
        self.progress_bar.pack()
        self.progress_bar.set(0)
        
    def update(self, text: str = None, progress: float = None):
        """Update text and/or progress."""
        if text is not None:
            self.label.configure(text=text)
        if progress is not None:
            self.progress_bar.set(progress)
            
    def start(self):
        """Start indeterminate progress."""
        self.progress_bar.start()
        
    def stop(self):
        """Stop indeterminate progress."""
        self.progress_bar.stop()