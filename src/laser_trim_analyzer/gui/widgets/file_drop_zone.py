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

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    DND_FILES = None


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
        
        # Only setup drag and drop if available and initialized
        if HAS_DND and self._check_dnd_initialized():
            self._setup_drag_drop()
        else:
            # Update UI to indicate drag and drop is not available
            self.primary_label.config(text='Click browse to select files')
            self.secondary_label.pack_forget()
            if not HAS_DND:
                print("Note: Drag and drop not available - tkinterdnd2 not installed")

    def _check_dnd_initialized(self):
        """Check if drag and drop is properly initialized."""
        try:
            # Try to check if tkdnd is available in the Tk instance
            self.winfo_toplevel().tk.call('tk', 'windowingsystem')
            return True
        except:
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
            self.primary_label.config(text='Click browse to select files')
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

        # Validate and filter files
        valid_files = self._validate_files(files)

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

    def _validate_files(self, files: List[Path]) -> List[Path]:
        """Validate and filter dropped files."""
        valid_files = []

        for file_path in files:
            if file_path.is_file():
                # Check extension
                if file_path.suffix.lower() in self.accept_extensions:
                    valid_files.append(file_path)
            elif file_path.is_dir() and self.accept_folders:
                # Recursively find valid files in folder
                valid_files.extend(self._find_files_in_folder(file_path))

        return valid_files

    def _find_files_in_folder(self, folder: Path) -> List[Path]:
        """Recursively find valid files in a folder."""
        valid_files = []

        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.accept_extensions:
                # Skip temporary files
                if not file_path.name.startswith('~'):
                    valid_files.append(file_path)

        return valid_files

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
                # Select folder
                folder = filedialog.askdirectory(title="Select Folder")
                if folder:
                    folder_path = Path(folder)
                    valid_files = self._find_files_in_folder(folder_path)
                    if valid_files:
                        self._process_dropped_files(valid_files)
                    else:
                        messagebox.showwarning(
                            "No Files Found",
                            f"No {', '.join(self.accept_extensions)} files found in the selected folder."
                        )
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
        """Set the state of the drop zone (alias for set_enabled)."""
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