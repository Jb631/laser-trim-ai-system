"""
Plot viewer widget for displaying analysis plots in the GUI.

This widget provides an integrated plot viewer with zoom, pan, and export capabilities.
"""

import customtkinter as ctk
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PlotViewerWidget(ctk.CTkFrame):
    """
    Widget for displaying plot images in the GUI with zoom and pan capabilities.
    
    Features:
    - Display PNG plot images
    - Zoom in/out functionality
    - Fit to window
    - Export/save plot
    - Thumbnail preview
    """
    
    def __init__(self, parent, **kwargs):
        """Initialize the plot viewer widget."""
        super().__init__(parent, **kwargs)
        
        # Internal state
        self.current_image_path: Optional[Path] = None
        self.original_image: Optional[Image.Image] = None
        self.displayed_image: Optional[Image.Image] = None
        self.ctk_image: Optional[ctk.CTkImage] = None
        self.zoom_level: float = 1.0
        self.min_zoom: float = 0.1
        self.max_zoom: float = 5.0
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        """Create the plot viewer UI."""
        # Main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Control bar at top
        self.control_bar = ctk.CTkFrame(self.main_container, height=40)
        self.control_bar.pack(fill='x', padx=5, pady=(5, 0))
        self.control_bar.pack_propagate(False)
        
        # Control buttons
        self.zoom_in_btn = ctk.CTkButton(
            self.control_bar,
            text="🔍+",
            width=40,
            height=30,
            command=self._zoom_in
        )
        self.zoom_in_btn.pack(side='left', padx=2)
        
        self.zoom_out_btn = ctk.CTkButton(
            self.control_bar,
            text="🔍-",
            width=40,
            height=30,
            command=self._zoom_out
        )
        self.zoom_out_btn.pack(side='left', padx=2)
        
        self.fit_btn = ctk.CTkButton(
            self.control_bar,
            text="Fit",
            width=60,
            height=30,
            command=self._fit_to_window
        )
        self.fit_btn.pack(side='left', padx=5)
        
        self.zoom_label = ctk.CTkLabel(
            self.control_bar,
            text="100%",
            font=ctk.CTkFont(size=12)
        )
        self.zoom_label.pack(side='left', padx=10)
        
        # Export button on the right
        self.export_btn = ctk.CTkButton(
            self.control_bar,
            text="Export",
            width=70,
            height=30,
            command=self._export_plot
        )
        self.export_btn.pack(side='right', padx=5)
        
        # Image display area with scrollable frame
        self.image_container = ctk.CTkScrollableFrame(
            self.main_container,
            corner_radius=0
        )
        self.image_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Image label
        self.image_label = ctk.CTkLabel(
            self.image_container,
            text="📊\n\nNo plot loaded\n\nComplete an analysis to generate plots",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.image_label.pack(expand=True)
        
        # Status bar
        self.status_bar = ctk.CTkFrame(self.main_container, height=25)
        self.status_bar.pack(fill='x', padx=5, pady=(0, 5))
        self.status_bar.pack_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            font=ctk.CTkFont(size=11),
            anchor='w'
        )
        self.status_label.pack(side='left', padx=5)
        
    def load_plot(self, plot_path: Path) -> bool:
        """
        Load and display a plot image.
        
        Args:
            plot_path: Path to the plot image file
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if not plot_path.exists():
                logger.warning(f"Plot file not found: {plot_path}")
                self._show_error("Plot file not found")
                return False
                
            # Load the image
            self.original_image = Image.open(plot_path)
            self.current_image_path = plot_path
            
            # Convert to RGB if necessary (for PNG with transparency)
            if self.original_image.mode in ('RGBA', 'LA'):
                # Create white background
                background = Image.new('RGB', self.original_image.size, 'white')
                if self.original_image.mode == 'RGBA':
                    background.paste(self.original_image, mask=self.original_image.split()[3])
                else:
                    background.paste(self.original_image, mask=self.original_image.split()[1])
                self.original_image = background
                
            # Reset zoom and display
            self.zoom_level = 1.0
            self._fit_to_window()
            
            # Update status
            self.status_label.configure(text=f"Loaded: {plot_path.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plot: {e}")
            self._show_error(f"Failed to load plot: {str(e)}")
            return False
            
    def _update_display(self):
        """Update the displayed image based on current zoom level."""
        if not self.original_image:
            return
        
        # Check if widget still exists
        if not self.winfo_exists():
            return
            
        try:
            # Calculate new size
            orig_width, orig_height = self.original_image.size
            new_width = int(orig_width * self.zoom_level)
            new_height = int(orig_height * self.zoom_level)
            
            # Resize image
            self.displayed_image = self.original_image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )
            
            # Keep previous CTkImage reference to prevent garbage collection
            # This prevents the "pyimage doesn't exist" error
            if hasattr(self, 'ctk_image') and self.ctk_image:
                if not hasattr(self, '_previous_ctk_images'):
                    self._previous_ctk_images = []
                self._previous_ctk_images.append(self.ctk_image)
                # Keep only last 2 images to prevent memory buildup
                if len(self._previous_ctk_images) > 2:
                    self._previous_ctk_images.pop(0)
            
            # Convert to CTkImage for proper scaling support
            self.ctk_image = ctk.CTkImage(
                light_image=self.displayed_image,
                dark_image=self.displayed_image,
                size=(new_width, new_height)
            )
            
            # Update label with CTkImage
            try:
                self.image_label.configure(image=self.ctk_image, text="")
                # Keep a reference to prevent garbage collection
                self.image_label.image = self.ctk_image
            except Exception as e:
                logger.error(f"Error configuring image label: {e}")
                # Try alternative approach
                self.image_label.image = self.ctk_image
                self.image_label.configure(image=self.ctk_image, text="")
            
            # Update zoom label
            zoom_percent = int(self.zoom_level * 100)
            self.zoom_label.configure(text=f"{zoom_percent}%")
            
        except Exception as e:
            logger.error(f"Failed to update display: {e}")
            
    def _zoom_in(self):
        """Zoom in by 25%."""
        if self.zoom_level < self.max_zoom:
            self.zoom_level = min(self.zoom_level * 1.25, self.max_zoom)
            self._update_display()
            
    def _zoom_out(self):
        """Zoom out by 25%."""
        if self.zoom_level > self.min_zoom:
            self.zoom_level = max(self.zoom_level / 1.25, self.min_zoom)
            self._update_display()
            
    def _fit_to_window(self):
        """Fit the image to the window size."""
        if not self.original_image:
            return
            
        try:
            # Get container size
            container_width = self.image_container.winfo_width()
            container_height = self.image_container.winfo_height()
            
            # Subtract some padding
            container_width -= 20
            container_height -= 20
            
            # If container not yet sized, use defaults
            if container_width <= 1:
                container_width = 800
            if container_height <= 1:
                container_height = 600
                
            # Calculate zoom to fit
            orig_width, orig_height = self.original_image.size
            zoom_width = container_width / orig_width
            zoom_height = container_height / orig_height
            
            # Use smaller zoom to fit both dimensions
            self.zoom_level = min(zoom_width, zoom_height, 1.0)  # Don't zoom beyond 100%
            self.zoom_level = max(self.zoom_level, self.min_zoom)
            
            self._update_display()
            
        except Exception as e:
            logger.error(f"Failed to fit to window: {e}")
            
    def _export_plot(self):
        """Export the current plot to a file."""
        if not self.original_image:
            self._show_error("No plot to export")
            return
            
        try:
            from tkinter import filedialog
            
            # Get save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ],
                initialfile=f"plot_export_{Path(self.current_image_path).stem}"
            )
            
            if file_path:
                # Save the original image
                self.original_image.save(file_path)
                self.status_label.configure(text=f"Exported to: {Path(file_path).name}")
                logger.info(f"Plot exported to: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to export plot: {e}")
            self._show_error(f"Failed to export: {str(e)}")
            
    def _show_error(self, message: str):
        """Show an error message in the viewer."""
        self.image_label.configure(
            image=None,
            text=f"❌\n\nError loading plot\n\n{message}",
            text_color="red"
        )
        self.status_label.configure(text="Error")
        
    def clear(self):
        """Clear the current plot display."""
        # Store current image before clearing to prevent premature garbage collection
        if hasattr(self, 'ctk_image') and self.ctk_image:
            if not hasattr(self, '_previous_ctk_images'):
                self._previous_ctk_images = []
            self._previous_ctk_images.append(self.ctk_image)
            # Keep only last 3 images to prevent memory buildup
            if len(self._previous_ctk_images) > 3:
                self._previous_ctk_images.pop(0)
        
        self.current_image_path = None
        self.original_image = None
        self.displayed_image = None
        self.ctk_image = None
        self.zoom_level = 1.0
        
        # Clear image reference but keep the widget
        if hasattr(self.image_label, 'image'):
            self.image_label.image = None
        
        # Configure label with text, but don't destroy the widget
        self.image_label.configure(
            image=None,
            text="📊\n\nNo plot loaded\n\nComplete an analysis to generate plots",
            text_color="gray"
        )
        self.zoom_label.configure(text="100%")
        self.status_label.configure(text="Ready")
        
    def refresh(self):
        """Refresh the display (e.g., after window resize)."""
        if self.original_image:
            # Refit to window on refresh
            self.after(100, self._fit_to_window)