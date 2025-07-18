"""
FileDropZone Widget - Drag and drop file selection

Provides an intuitive drag-and-drop interface for file selection
with visual feedback and multi-file support.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Optional, Callable, Set
import os
import threading
import time
import tempfile
import shutil

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    DND_FILES = None

# Import security utilities
try:
    from laser_trim_analyzer.core.security import (
        SecurityValidator, SecureFileProcessor,
        get_security_validator
    )
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False


class FileDropZone(ttk.Frame):
    """
    Modern drag-and-drop zone for file selection.

    Features:
    - Drag and drop support for files and folders (if tkinterdnd2 available)
    - Visual feedback during drag operations
    - File type filtering
    - Progress indication for large operations
    - Callback support for file events
    """

    def __init__(
            self,
            parent,
            accept_extensions: List[str] = None,
            accept_folders: bool = True,
            on_files_dropped: Optional[Callable[[List[Path]], None]] = None,
            height: int = 200,
            **kwargs
    ):
        """
        Initialize FileDropZone.

        Args:
            parent: Parent widget
            accept_extensions: List of accepted file extensions (e.g., ['.xlsx', '.xls'])
            accept_folders: Whether to accept folder drops
            on_files_dropped: Callback when files are dropped
            height: Zone height in pixels
        """
        super().__init__(parent, **kwargs)

        self.accept_extensions = accept_extensions or ['.xlsx', '.xls']
        self.accept_folders = accept_folders
        self.on_files_dropped = on_files_dropped
        self.dropped_files: List[Path] = []

        # State
        self.is_dragging = False

        # Colors
        self.colors = {
            'normal_bg': '#f8f9fa',
            'normal_border': '#dee2e6',
            'hover_bg': '#e3f2fd',
            'hover_border': '#2196f3',
            'accept_bg': '#e8f5e9',
            'accept_border': '#4caf50',
            'reject_bg': '#ffebee',
            'reject_border': '#f44336',
            'text_normal': '#6c757d',
            'text_hover': '#2196f3'
        }

        self._setup_ui(height)
        
        # Setup drag and drop if available
        if HAS_DND:
            try:
                self._setup_drag_drop()
            except Exception as e:
                # Fallback if drag and drop setup fails
                self.logger = parent.logger if hasattr(parent, 'logger') else None
                if self.logger:
                    self.logger.warning(f"Drag and drop setup failed: {e}")
                self.primary_label.configure(text='Click browse to select files')
                self.secondary_label.pack_forget()
        else:
            # Update UI to indicate drag and drop is not available
            error_msg = "Drag and drop requires tkinterdnd2. Install with: pip install tkinterdnd2"
            self.primary_label.configure(text='Click browse to select files')
            self.secondary_label.pack_forget()
            # Log warning but don't crash the app
            if self.logger:
                self.logger.warning(error_msg)
            # Show in UI that drag-drop is unavailable but allow browse functionality
            self.secondary_label.configure(text="(Drag & drop unavailable - use browse button)")
            self.secondary_label.pack()

    def _check_dnd_initialized(self):
        """Check if drag and drop is properly initialized."""
        try:
            # Check if we're using a TkinterDnD window
            root = self.winfo_toplevel()
            if hasattr(root, 'TkdndVersion'):
                return True
            # Try to initialize tkdnd
            root.tk.call('package', 'require', 'tkdnd')
            return True
        except Exception as e:
            return False

    def _setup_ui(self, height: int):
        """Set up the UI components."""
        # Main drop frame
        self.drop_frame = tk.Frame(
            self,
            bg=self.colors['normal_bg'],
            relief='solid',
            borderwidth=2,
            highlightbackground=self.colors['normal_border'],
            highlightthickness=2
        )
        self.drop_frame.pack(fill='both', expand=True)

        # Set minimum height
        self.drop_frame.configure(height=height)

        # Center container
        center_frame = tk.Frame(self.drop_frame, bg=self.colors['normal_bg'])
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        # Icon (using Unicode for simplicity)
        self.icon_label = tk.Label(
            center_frame,
            text='📁',
            font=('Segoe UI', 48),
            bg=self.colors['normal_bg'],
            fg=self.colors['text_normal']
        )
        self.icon_label.pack()

        # Primary text
        self.primary_label = tk.Label(
            center_frame,
            text='Drag and drop files here',
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['normal_bg'],
            fg=self.colors['text_normal']
        )
        self.primary_label.pack(pady=(10, 5))

        # Secondary text
        self.secondary_label = tk.Label(
            center_frame,
            text='or',
            font=('Segoe UI', 11),
            bg=self.colors['normal_bg'],
            fg=self.colors['text_normal']
        )
        self.secondary_label.pack()

        # Browse button
        self.browse_button = ttk.Button(
            center_frame,
            text='Browse Files',
            command=self._browse_files,
            style='Accent.TButton'
        )
        self.browse_button.pack(pady=10)

        # Accepted types text
        extensions_text = f"Accepted: {', '.join(self.accept_extensions)}"
        if self.accept_folders:
            extensions_text += " and folders"

        self.types_label = tk.Label(
            center_frame,
            text=extensions_text,
            font=('Segoe UI', 9),
            bg=self.colors['normal_bg'],
            fg=self.colors['text_normal']
        )
        self.types_label.pack(pady=(10, 0))

        # File count label (hidden initially)
        self.count_label = tk.Label(
            self.drop_frame,
            text='',
            font=('Segoe UI', 10),
            bg=self.colors['normal_bg'],
            fg=self.colors['text_normal']
        )

    def _setup_drag_drop(self):
        """Set up drag and drop functionality."""
        try:
            # Make the drop frame a drop target
            self.drop_frame.drop_target_register(DND_FILES)

            # Bind drag and drop events
            self.drop_frame.dnd_bind('<<DropEnter>>', self._on_drag_enter)
            self.drop_frame.dnd_bind('<<DropPosition>>', self._on_drag_motion)
            self.drop_frame.dnd_bind('<<DropLeave>>', self._on_drag_leave)
            self.drop_frame.dnd_bind('<<Drop>>', self._on_drop)
        except Exception as e:
            print(f"Could not enable drag and drop: {e}")
            # Update UI to indicate drag and drop is not available
            self.primary_label.configure(text='Click browse to select files')
            self.secondary_label.pack_forget()

    def _on_drag_enter(self, event):
        """Handle drag enter event."""
        self.is_dragging = True
        self._update_appearance('hover')
        return event.action

    def _on_drag_motion(self, event):
        """Handle drag motion event."""
        # Could add position-based feedback here
        return event.action

    def _on_drag_leave(self, event):
        """Handle drag leave event."""
        self.is_dragging = False
        self._update_appearance('normal')

    def _on_drop(self, event):
        """Handle file drop event."""
        self.is_dragging = False

        # Parse dropped files
        files = self._parse_drop_data(event.data)

        # Check if any folders are dropped
        has_folders = any(f.is_dir() for f in files if f.exists())
        
        if has_folders and self.accept_folders:
            # Handle async folder processing
            self._process_dropped_files_async(files)
        else:
            # Validate and filter files synchronously (for individual files)
            valid_files = self._validate_files_sync(files)

            if valid_files:
                self._update_appearance('accept')
                self.after(500, lambda: self._update_appearance('normal'))
                # Process files
                self._process_dropped_files(valid_files)
            else:
                self._update_appearance('reject')
                self.after(500, lambda: self._update_appearance('normal'))

        return event.action

    def _parse_drop_data(self, data: str) -> List[Path]:
        """Parse the dropped data into file paths."""
        # Handle different formats
        files = []

        # Split by newlines and spaces (different platforms)
        parts = data.replace('{', '').replace('}', '').split()

        for part in parts:
            if part:
                try:
                    path = Path(part)
                    if path.exists():
                        files.append(path)
                except:
                    pass

        return files

    def _process_dropped_files_async(self, files: List[Path]):
        """Process dropped files that may include folders asynchronously."""
        # Update UI to scanning state
        self._update_appearance('hover')
        self.primary_label.configure(text='Processing dropped items...')
        self.secondary_label.configure(text='Please wait while we scan folders')
        self.browse_button.configure(state='disabled')
        
        def process_files():
            try:
                valid_files = []
                total_checked = 0
                
                for file_path in files:
                    if file_path.is_file():
                        # Check extension for individual files
                        if file_path.suffix.lower() in self.accept_extensions:
                            valid_files.append(file_path)
                    elif file_path.is_dir() and self.accept_folders:
                        # Recursively find valid files in folder
                        for found_file in file_path.rglob('*'):
                            if found_file.is_file() and found_file.suffix.lower() in self.accept_extensions:
                                # Skip temporary files and hidden files
                                if not found_file.name.startswith('~') and not found_file.name.startswith('.'):
                                    # Security check for files from folders
                                    security_validator = None
                                    if HAS_SECURITY:
                                        try:
                                            security_validator = get_security_validator()
                                        except:
                                            pass
                                    
                                    if security_validator:
                                        try:
                                            path_result = security_validator.validate_input(
                                                found_file,
                                                'file_path',
                                                {
                                                    'allowed_extensions': self.accept_extensions,
                                                    'check_extension': True
                                                }
                                            )
                                            
                                            if path_result.is_safe and not path_result.threats_detected:
                                                valid_files.append(Path(path_result.sanitized_value))
                                        except:
                                            # Fall back to adding if security check fails
                                            valid_files.append(found_file)
                                    else:
                                        valid_files.append(found_file)
                            
                            # Update progress every 100 files
                            total_checked += 1
                            if total_checked % 100 == 0:
                                # Capture the current count value
                                current_count = total_checked
                                self.after(0, lambda count=current_count: 
                                         self.secondary_label.configure(text=f'Scanned {count} files...'))
                
                # Call completion handler on main thread
                self.after(0, self._handle_async_drop_complete, valid_files, None)
                
            except Exception as e:
                # Call completion handler with error
                self.after(0, self._handle_async_drop_complete, [], str(e))
        
        # Start processing thread
        thread = threading.Thread(target=process_files, daemon=True)
        thread.start()

    def _handle_async_drop_complete(self, valid_files: List[Path], error: str):
        """Handle completion of async drop processing."""
        # Re-enable UI
        self.browse_button.configure(state='normal')
        
        if error:
            self._update_appearance('reject')
            self.primary_label.configure(text='Error processing dropped items')
            self.secondary_label.configure(text=f'Error: {error}')
            self.after(2000, lambda: self._reset_ui_text())
        elif valid_files:
            self._update_appearance('accept')
            self.primary_label.configure(text=f'Found {len(valid_files)} files!')
            self.secondary_label.configure(text='Files ready for processing')
            self._process_dropped_files(valid_files)
            self.after(2000, lambda: self._reset_ui_text())
        else:
            self._update_appearance('reject')
            self.primary_label.configure(text='No valid files found')
            self.secondary_label.configure(text='Try dropping files with supported extensions')
            self.after(2000, lambda: self._reset_ui_text())

    def _reset_ui_text(self):
        """Reset UI text to default state."""
        self._update_appearance('normal')
        self.primary_label.configure(text='Drag and drop files here')
        self.secondary_label.configure(text='or click browse to select files')

    def _validate_files_sync(self, files: List[Path]) -> List[Path]:
        """Validate and filter dropped files synchronously with security checks."""
        valid_files = []
        
        # Get security validator if available
        security_validator = None
        if HAS_SECURITY:
            try:
                security_validator = get_security_validator()
            except:
                pass

        for file_path in files:
            if file_path.is_file():
                # Security validation if available
                if security_validator:
                    try:
                        # Validate file path for security threats
                        path_result = security_validator.validate_input(
                            file_path,
                            'file_path',
                            {
                                'allowed_extensions': self.accept_extensions,
                                'check_extension': True
                            }
                        )
                        
                        if not path_result.is_safe:
                            print(f"Security validation failed for {file_path}: {path_result.validation_errors}")
                            continue
                            
                        if path_result.threats_detected:
                            print(f"Security threat detected in {file_path}: {path_result.threats_detected}")
                            continue
                            
                        # Use sanitized path
                        file_path = Path(path_result.sanitized_value)
                    except Exception as e:
                        print(f"Security validation error for {file_path}: {e}")
                        # Continue with basic validation if security check fails
                
                # Basic extension check
                if file_path.suffix.lower() in self.accept_extensions:
                    # Additional file size check
                    try:
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        max_size_mb = 100  # Default max size
                        
                        if file_size_mb > max_size_mb:
                            print(f"File too large: {file_path.name} ({file_size_mb:.1f}MB)")
                            continue
                    except:
                        pass
                    
                    valid_files.append(file_path)

        return valid_files

    def _validate_files(self, files: List[Path]) -> List[Path]:
        """Validate and filter dropped files (deprecated - use _validate_files_sync)."""
        return self._validate_files_sync(files)

    def _find_files_in_folder(self, folder: Path) -> List[Path]:
        """Recursively find valid files in a folder (synchronous)."""
        valid_files = []

        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.accept_extensions:
                # Skip temporary files
                if not file_path.name.startswith('~'):
                    valid_files.append(file_path)

        return valid_files

    def _find_files_in_folder_async(self, folder: Path, callback):
        """Asynchronously find valid files in a folder."""
        self._update_appearance('hover')
        self.primary_label.configure(text='Scanning folder...')
        self.secondary_label.configure(text='Please wait while we discover files')
        self.browse_button.configure(state='disabled')
        
        def discover_files():
            try:
                valid_files = []
                total_checked = 0
                
                for file_path in folder.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in self.accept_extensions:
                        # Skip temporary files
                        if not file_path.name.startswith('~'):
                            valid_files.append(file_path)
                    
                    # Update progress every 100 files
                    total_checked += 1
                    if total_checked % 100 == 0:
                        # Capture the current count value
                        current_count = total_checked
                        self.after(0, lambda count=current_count: 
                                 self.secondary_label.configure(text=f'Scanned {count} files...'))
                
                # Call callback on main thread with results
                self.after(0, callback, valid_files, None)
                
            except Exception as e:
                # Call callback with error
                self.after(0, callback, [], str(e))
        
        # Start discovery thread
        thread = threading.Thread(target=discover_files, daemon=True)
        thread.start()

    def _browse_files(self):
        """Open file browser dialog."""
        if self.accept_folders:
            # Ask user what they want to select
            result = messagebox.askyesnocancel(
                "Select Files or Folder",
                "Do you want to select individual files?\n\n"
                "Yes - Select files\n"
                "No - Select folder\n"
                "Cancel - Cancel operation"
            )

            if result is True:
                # Select files
                files = filedialog.askopenfilenames(
                    title="Select Excel Files",
                    filetypes=[
                        ("Excel files", " ".join(f"*{ext}" for ext in self.accept_extensions)),
                        ("All files", "*.*")
                    ]
                )
                if files:
                    valid_files = [Path(f) for f in files]
                    self._process_dropped_files(valid_files)

            elif result is False:
                # Select folder with async discovery
                folder = filedialog.askdirectory(title="Select Folder")
                if folder:
                    folder_path = Path(folder)
                    
                    def on_discovery_complete(valid_files, error):
                        # Re-enable UI
                        self.browse_button.configure(state='normal')
                        self._update_appearance('normal')
                        
                        if error:
                            self.primary_label.configure(text='Error scanning folder')
                            self.secondary_label.configure(text=f'Error: {error}')
                            messagebox.showerror("Folder Scan Error", f"Could not scan folder:\n{error}")
                        elif valid_files:
                            self.primary_label.configure(text='Drag and drop files here')
                            self.secondary_label.configure(text='or click browse to select files')
                            self._process_dropped_files(valid_files)
                            messagebox.showinfo(
                                "Folder Scan Complete", 
                                f"Found {len(valid_files)} {', '.join(self.accept_extensions)} files\n"
                                f"in {folder_path.name}"
                            )
                        else:
                            self.primary_label.configure(text='No files found')
                            self.secondary_label.configure(text='Try selecting a different folder')
                            messagebox.showwarning(
                                "No Files Found",
                                f"No {', '.join(self.accept_extensions)} files found in the selected folder."
                            )
                    
                    # Start async discovery
                    self._find_files_in_folder_async(folder_path, on_discovery_complete)
        else:
            # Just select files
            files = filedialog.askopenfilenames(
                title="Select Files",
                filetypes=[
                    ("Accepted files", " ".join(f"*{ext}" for ext in self.accept_extensions)),
                    ("All files", "*.*")
                ]
            )
            if files:
                valid_files = [Path(f) for f in files]
                self._process_dropped_files(valid_files)

    def _process_dropped_files(self, files: List[Path]):
        """Process the dropped files."""
        # Remove duplicates
        unique_files = []
        seen = set()

        for file in files:
            if file not in seen:
                seen.add(file)
                unique_files.append(file)

        self.dropped_files.extend(unique_files)

        # Update UI
        self._update_file_count()

        # Callback - convert Path objects to strings
        if self.on_files_dropped:
            self.on_files_dropped([str(f) for f in unique_files])

    def _update_appearance(self, state: str):
        """Update the visual appearance based on state."""
        if state == 'normal':
            bg_color = self.colors['normal_bg']
            border_color = self.colors['normal_border']
            text_color = self.colors['text_normal']
            icon_text = '📁'
        elif state == 'hover':
            bg_color = self.colors['hover_bg']
            border_color = self.colors['hover_border']
            text_color = self.colors['text_hover']
            icon_text = '📂'
        elif state == 'accept':
            bg_color = self.colors['accept_bg']
            border_color = self.colors['accept_border']
            text_color = self.colors['accept_border']
            icon_text = '✅'
        elif state == 'reject':
            bg_color = self.colors['reject_bg']
            border_color = self.colors['reject_border']
            text_color = self.colors['reject_border']
            icon_text = '❌'
        elif state == 'processing':
            bg_color = '#fff3cd'  # Light yellow
            border_color = '#ffc107'  # Warning yellow
            text_color = '#856404'  # Dark yellow
            icon_text = '⚙️'
        else:
            return

        # Update colors
        self.drop_frame.configure(
            bg=bg_color,
            highlightbackground=border_color
        )

        # Update all child widgets
        for widget in self.drop_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.configure(bg=bg_color)
                for child in widget.winfo_children():
                    if isinstance(child, tk.Label):
                        child.configure(bg=bg_color, fg=text_color)

        # Update icon
        self.icon_label.configure(text=icon_text)

    def _update_file_count(self):
        """Update the file count display."""
        count = len(self.dropped_files)
        if count > 0:
            self.count_label.configure(text=f"{count} file{'s' if count != 1 else ''} selected")
            self.count_label.place(x=10, y=10)
        else:
            self.count_label.place_forget()

    def clear_files(self):
        """Clear all dropped files."""
        self.dropped_files.clear()
        self._update_file_count()
        self._update_appearance('normal')

    def get_files(self) -> List[Path]:
        """Get the list of dropped files."""
        return self.dropped_files.copy()

    def set_enabled(self, enabled: bool):
        """Enable or disable the drop zone."""
        if enabled:
            self.browse_button.configure(state='normal')
            if HAS_DND and self._check_dnd_initialized():
                self._setup_drag_drop()
            self._update_appearance('normal')
        else:
            self.browse_button.configure(state='disabled')
            # Unbind drag and drop events if available
            if HAS_DND and self._check_dnd_initialized():
                try:
                    self.drop_frame.drop_target_unregister()
                except:
                    pass
            self._update_appearance('normal')

    def set_state(self, state: str):
        """Set the state of the drop zone."""
        if state == 'normal':
            self.set_enabled(True)
            self._update_appearance('normal')
        elif state == 'disabled':
            self.set_enabled(False)
            self._update_appearance('normal')
        elif state == 'processing':
            # Keep enabled but show processing state
            self.browse_button.configure(state='disabled')  # Disable browse during processing
            self._update_appearance('processing')
            # Update text to show processing
            self.primary_label.configure(text='Processing files...')
            self.secondary_label.configure(text='Analysis in progress')
        else:
            self.set_enabled(state == 'normal')


# Example usage
if __name__ == "__main__":
    # Create test window
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        
    root.title("FileDropZone Demo")
    root.geometry("600x400")


    # Callback function
    def on_files_dropped(files: List[str]):
        print(f"Files dropped: {[Path(f).name for f in files]}")


    # Create drop zone
    drop_zone = FileDropZone(
        root,
        accept_extensions=['.xlsx', '.xls', '.csv'],
        accept_folders=True,
        on_files_dropped=on_files_dropped
    )
    drop_zone.pack(fill='both', expand=True, padx=20, pady=20)

    # Buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=10)

    ttk.Button(
        button_frame,
        text="Clear Files",
        command=drop_zone.clear_files
    ).pack(side='left', padx=5)

    ttk.Button(
        button_frame,
        text="Get Files",
        command=lambda: print(drop_zone.get_files())
    ).pack(side='left', padx=5)

    root.mainloop()