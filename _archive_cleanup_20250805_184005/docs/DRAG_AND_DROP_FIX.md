# Drag-and-Drop Functionality Fix

## Summary

Fixed the drag-and-drop functionality in the Laser Trim Analyzer V2 application by:

1. **Modified the main window** to support TkinterDnD2 when available
2. **Added FileDropZone** to the batch processing page for drag-and-drop file selection
3. **Implemented proper fallback** when tkinterdnd2 is not available

## Changes Made

### 1. Main Window Initialization (`ctk_main_window.py`)

Modified the main window to use TkinterDnD as the base class when available:

```python
# Try to import TkinterDnD for drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    TkinterDnD = None

# Create base class depending on drag-and-drop availability
if HAS_DND:
    class CTkMainWindowBase(TkinterDnD.Tk):
        """Base window with drag-and-drop support"""
        pass
else:
    class CTkMainWindowBase(ctk.CTk):
        """Base window without drag-and-drop support"""
        pass
```

This ensures the main window supports drag-and-drop operations when tkinterdnd2 is installed.

### 2. Batch Processing Page (`batch_processing_page.py`)

Added FileDropZone widget to the file selection section:

```python
# Create FileDropZone for drag-and-drop functionality
try:
    self.drop_zone = FileDropZone(
        self.file_selection_frame,
        accept_extensions=['.xlsx', '.xls'],
        accept_folders=True,
        on_files_dropped=self._on_files_dropped,
        height=120
    )
    self.drop_zone.pack(fill='x', padx=15, pady=(5, 10))
except Exception as e:
    # Fallback if drop zone fails
    placeholder_label = ctk.CTkLabel(
        self.file_selection_frame,
        text="Drag-and-drop not available. Use buttons below to select files.",
        font=ctk.CTkFont(size=12),
        text_color="gray"
    )
    placeholder_label.pack(pady=10)
```

### 3. File Drop Handler

Implemented `_on_files_dropped` method to handle dropped files:

```python
def _on_files_dropped(self, file_paths: List[str]):
    """Handle files dropped on the drop zone."""
    # Convert string paths to Path objects
    path_objects = [Path(fp) for fp in file_paths]
    
    # Validate and add Excel files
    validated_files = []
    for path_obj in path_objects:
        # Only process Excel files
        if path_obj.suffix.lower() in ['.xlsx', '.xls', '.xlsm']:
            validated_files.append(path_obj)
    
    if validated_files:
        # Add to existing selection
        for file in validated_files:
            if file not in self.selected_files:
                self.selected_files.append(file)
        
        self._update_file_display()
        self._update_batch_status("Files Added", "orange")
```

## How It Works

1. **With tkinterdnd2 installed**: 
   - Users can drag Excel files or folders directly onto the drop zone
   - Visual feedback is provided during drag operations
   - Multiple files can be dropped at once
   - Folders are scanned recursively for Excel files

2. **Without tkinterdnd2**:
   - The drop zone shows a message indicating drag-and-drop is not available
   - Users can still use the "Select Files" and "Select Folder" buttons
   - All other functionality remains the same

## Installation Requirements

For drag-and-drop to work, ensure tkinterdnd2 is installed:

```bash
pip install tkinterdnd2
```

The requirement is already listed in `requirements.txt`.

## Benefits

1. **Improved User Experience**: Users can quickly add files by dragging them from file explorer
2. **Batch Operations**: Users can drag entire folders to process all Excel files within
3. **Visual Feedback**: The drop zone provides clear visual feedback during drag operations
4. **Graceful Fallback**: The application works normally even without drag-and-drop support

## Testing

To verify drag-and-drop is working:

1. Launch the application
2. Navigate to the Batch Processing page
3. You should see a drop zone with the text "Drag and drop files here"
4. Drag Excel files from your file explorer onto the zone
5. The files should be added to the processing list

If drag-and-drop is not available, you'll see a message indicating to use the buttons instead.