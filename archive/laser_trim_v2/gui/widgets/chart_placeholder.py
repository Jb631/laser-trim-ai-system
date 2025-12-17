"""
Chart Placeholder Widget for Laser Trim Analyzer

Provides a polished placeholder for empty charts that shows
an informative message or icon while data is being loaded or
when no data is available.
"""

import customtkinter as ctk
from typing import Optional, Literal

class ChartPlaceholder(ctk.CTkFrame):
    """A placeholder widget for empty charts with customizable messages."""
    
    def __init__(
        self,
        parent,
        width: int = 400,
        height: int = 300,
        message: str = "No data available",
        icon: str = "📊",
        instruction: Optional[str] = None,
        style: Literal["info", "waiting", "empty", "error"] = "empty",
        **kwargs
    ):
        """
        Initialize chart placeholder.
        
        Args:
            parent: Parent widget
            width: Width of placeholder
            height: Height of placeholder  
            message: Main message to display
            icon: Icon/emoji to display (default: chart icon)
            instruction: Optional instruction text
            style: Visual style (info, waiting, empty, error)
        """
        # Set appropriate background color based on style
        fg_colors = {
            "info": ("gray90", "gray20"),
            "waiting": ("gray95", "gray15"),
            "empty": ("gray85", "gray25"),
            "error": ("#ffebee", "#5a1e1e")
        }
        
        kwargs["fg_color"] = fg_colors.get(style, fg_colors["empty"])
        kwargs["width"] = width
        kwargs["height"] = height
        
        super().__init__(parent, **kwargs)
        
        self.message = message
        self.icon = icon
        self.instruction = instruction
        self.style = style
        
        self._create_content()
        
    def _create_content(self):
        """Create placeholder content."""
        # Center frame for content
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Icon
        icon_size = 48 if self.style == "waiting" else 64
        self.icon_label = ctk.CTkLabel(
            content_frame,
            text=self.icon,
            font=ctk.CTkFont(size=icon_size)
        )
        self.icon_label.pack(pady=(0, 10))
        
        # Main message
        text_colors = {
            "info": ("gray40", "gray60"),
            "waiting": ("gray50", "gray50"),
            "empty": ("gray45", "gray55"),
            "error": ("#d32f2f", "#f44336")
        }
        
        text_color = text_colors.get(self.style, text_colors["empty"])
        
        self.message_label = ctk.CTkLabel(
            content_frame,
            text=self.message,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=text_color
        )
        self.message_label.pack(pady=(0, 5))
        
        # Instruction text (if provided)
        if self.instruction:
            self.instruction_label = ctk.CTkLabel(
                content_frame,
                text=self.instruction,
                font=ctk.CTkFont(size=12),
                text_color=text_color,
                wraplength=300
            )
            self.instruction_label.pack(pady=(0, 10))
            
        # Add loading animation for waiting state
        if self.style == "waiting":
            self.loading_label = ctk.CTkLabel(
                content_frame,
                text="Loading...",
                font=ctk.CTkFont(size=12),
                text_color=text_color
            )
            self.loading_label.pack()
            self._animate_loading()
            
    def _animate_loading(self):
        """Simple loading animation for waiting state."""
        if self.style == "waiting" and hasattr(self, 'loading_label'):
            current = self.loading_label.cget("text")
            dots = current.count('.')
            new_dots = (dots % 3) + 1
            self.loading_label.configure(text=f"Loading{'.' * new_dots}")
            self.after(500, self._animate_loading)
            
    def update_content(
        self,
        message: Optional[str] = None,
        icon: Optional[str] = None,
        instruction: Optional[str] = None,
        style: Optional[str] = None
    ):
        """Update placeholder content."""
        if message is not None:
            self.message = message
            self.message_label.configure(text=message)
            
        if icon is not None:
            self.icon = icon
            self.icon_label.configure(text=icon)
            
        if instruction is not None:
            self.instruction = instruction
            if hasattr(self, 'instruction_label'):
                self.instruction_label.configure(text=instruction)
            elif instruction:
                # Create instruction label if it doesn't exist
                self.instruction_label = ctk.CTkLabel(
                    self.message_label.master,
                    text=instruction,
                    font=ctk.CTkFont(size=12),
                    wraplength=300
                )
                self.instruction_label.pack(pady=(0, 10))
                
        if style is not None and style != self.style:
            self.style = style
            # Recreate content with new style
            for widget in self.winfo_children():
                widget.destroy()
            self._create_content()


# Convenience functions for common placeholder types
def create_chart_placeholder(parent, **kwargs) -> ChartPlaceholder:
    """Create a standard chart placeholder."""
    return ChartPlaceholder(
        parent,
        message="No data to display",
        icon="📊",
        instruction="Complete an analysis to see chart data",
        style="empty",
        **kwargs
    )

def create_loading_placeholder(parent, message: str = "Loading data...", **kwargs) -> ChartPlaceholder:
    """Create a loading placeholder."""
    return ChartPlaceholder(
        parent,
        message=message,
        icon="⏳",
        style="waiting",
        **kwargs
    )

def create_error_placeholder(parent, error_message: str, **kwargs) -> ChartPlaceholder:
    """Create an error placeholder."""
    return ChartPlaceholder(
        parent,
        message="Error loading chart",
        icon="❌",
        instruction=error_message,
        style="error",
        **kwargs
    )

def create_action_placeholder(parent, action: str, **kwargs) -> ChartPlaceholder:
    """Create a placeholder that prompts for action."""
    return ChartPlaceholder(
        parent,
        message="Action Required",
        icon="👆",
        instruction=action,
        style="info",
        **kwargs
    )