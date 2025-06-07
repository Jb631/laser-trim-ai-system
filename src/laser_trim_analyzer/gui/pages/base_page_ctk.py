"""
Base Page Class for CustomTkinter GUI Pages

Provides common functionality for all application pages including responsive design.
"""

import customtkinter as ctk
from typing import Optional, Dict, Any, Callable
import logging


class ResponsiveFrame(ctk.CTkFrame):
    """
    Frame that provides responsive layout capabilities.
    
    Features:
    - Automatic layout adjustment based on window size
    - Breakpoint-based responsive behavior
    - Dynamic column adjustment for grid layouts
    """
    
    def __init__(self, parent, breakpoints=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Default responsive breakpoints (width in pixels)
        self.breakpoints = breakpoints or {
            'small': 800,
            'medium': 1200,
            'large': 1600
        }
        
        self.current_size_class = 'large'
        self.layout_callbacks = []
        
        # Bind to window configuration changes
        self.bind('<Configure>', self._on_configure)
        
    def add_layout_callback(self, callback: Callable[[str], None]):
        """Add callback to be called when layout changes."""
        self.layout_callbacks.append(callback)
        
    def _on_configure(self, event):
        """Handle window resize events."""
        # Only respond to size changes for this widget
        if event.widget != self:
            return
            
        width = self.winfo_width()
        new_size_class = self._get_size_class(width)
        
        if new_size_class != self.current_size_class:
            self.current_size_class = new_size_class
            self._notify_layout_change()
            
    def _get_size_class(self, width: int) -> str:
        """Determine size class based on width."""
        if width <= self.breakpoints['small']:
            return 'small'
        elif width <= self.breakpoints['medium']:
            return 'medium'
        else:
            return 'large'
            
    def _notify_layout_change(self):
        """Notify all callbacks of layout change."""
        for callback in self.layout_callbacks:
            try:
                callback(self.current_size_class)
            except Exception as e:
                logging.getLogger(__name__).error(f"Layout callback error: {e}")
                
    def get_responsive_columns(self, items_count: int) -> int:
        """Get number of columns for responsive grid layout."""
        if self.current_size_class == 'small':
            return min(1, items_count)
        elif self.current_size_class == 'medium':
            return min(2, items_count)
        else:
            return min(3, items_count)
            
    def get_responsive_padding(self) -> Dict[str, int]:
        """Get responsive padding values."""
        if self.current_size_class == 'small':
            return {'padx': 5, 'pady': 5}
        elif self.current_size_class == 'medium':
            return {'padx': 10, 'pady': 10}
        else:
            return {'padx': 15, 'pady': 15}


class BasePage(ResponsiveFrame):
    """
    Base class for all application pages using CustomTkinter.

    Provides common functionality like show/hide, refresh, access to main window,
    and responsive design capabilities.
    """

    def __init__(self, parent: ctk.CTkFrame, main_window: Any, **kwargs):
        """
        Initialize base page.

        Args:
            parent: Parent frame
            main_window: Reference to main window for accessing services
        """
        super().__init__(parent, **kwargs)
        self.main_window = main_window
        self.logger = logging.getLogger(self.__class__.__name__)

        # Page state
        self.is_visible = False
        self.needs_refresh = True
        self._stop_requested = False  # For processing control

        # Common style configuration for CustomTkinter
        self.configure(corner_radius=0)
        
        # Responsive layout support
        self.add_layout_callback(self._handle_responsive_layout)

        # Initialize the page
        try:
            self._create_page()
        except Exception as e:
            self.logger.error(f"Error creating page {self.__class__.__name__}: {e}", exc_info=True)
            # Create error display
            self._create_error_page(str(e))

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

    def _create_page(self):
        """Create page content. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_page")

    def _handle_responsive_layout(self, size_class: str):
        """Handle responsive layout changes. Override in subclasses."""
        self.logger.debug(f"Layout changed to {size_class}")
        
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
        self.on_hide()

    def refresh(self):
        """Refresh page content. Override in subclasses."""
        pass

    def on_show(self):
        """Called when page is shown. Override in subclasses."""
        pass

    def on_hide(self):
        """Called when page is hidden. Override in subclasses."""
        pass

    def mark_needs_refresh(self):
        """Mark that the page needs refresh on next show."""
        self.needs_refresh = True
        
    def request_stop_processing(self):
        """Request that any ongoing processing be stopped."""
        self._stop_requested = True
        self.logger.info(f"{self.__class__.__name__}: Stop processing requested")
        
    def reset_stop_request(self):
        """Reset the stop processing flag."""
        self._stop_requested = False
        
    def is_stop_requested(self) -> bool:
        """Check if processing should be stopped."""
        return self._stop_requested
        
    def create_responsive_grid(self, parent, widgets, columns_config=None):
        """
        Create a responsive grid layout.
        
        Args:
            parent: Parent frame
            widgets: List of widgets to arrange
            columns_config: Optional dict with size_class -> column_count mapping
        """
        if not widgets:
            return
            
        # Default column configuration
        default_config = {
            'small': 1,
            'medium': 2,
            'large': 3
        }
        columns_config = columns_config or default_config
        
        # Get current column count
        columns = columns_config.get(self.current_size_class, 3)
        columns = min(columns, len(widgets))
        
        # Configure grid weights
        for i in range(columns):
            parent.columnconfigure(i, weight=1)
            
        # Arrange widgets
        for i, widget in enumerate(widgets):
            row = i // columns
            col = i % columns
            padding = self.get_responsive_padding()
            widget.grid(row=row, column=col, sticky='ew', **padding)

    @property
    def db_manager(self):
        """Get database manager from main window."""
        return getattr(self.main_window, 'db_manager', None)

    @property
    def app_config(self):
        """Get configuration from main window."""
        return getattr(self.main_window, 'config', None)

    @property
    def colors(self):
        """Get color scheme from main window."""
        # CustomTkinter doesn't use colors dict, return empty
        return {}

    def cleanup(self):
        """Cleanup resources when page is destroyed."""
        # Stop any processing
        self.request_stop_processing()
        self.logger.info(f"{self.__class__.__name__} cleanup completed")