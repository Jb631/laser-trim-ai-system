"""
Hover Fix Module

Fixes hover effects glitching and shifting issues in the GUI
by providing stable hover behavior.
"""

import customtkinter as ctk
from typing import Optional, Callable


class StableHoverMixin:
    """Mixin class to provide stable hover behavior without glitching."""
    
    def setup_stable_hover(self, 
                          normal_color: str,
                          hover_color: str,
                          transition_time: int = 100):
        """Setup stable hover effects without glitching."""
        self._normal_color = normal_color
        self._hover_color = hover_color
        self._transition_time = transition_time
        self._hover_job = None
        self._is_hovering = False
        
        # Bind hover events
        self.bind("<Enter>", lambda e: self._on_hover_enter())
        self.bind("<Leave>", lambda e: self._on_hover_leave())
        
    def _on_hover_enter(self):
        """Handle mouse enter with debouncing."""
        self._is_hovering = True
        # Cancel any pending leave transition
        if self._hover_job:
            self.after_cancel(self._hover_job)
            self._hover_job = None
        
        # Apply hover color immediately
        try:
            self.configure(fg_color=self._hover_color)
        except:
            pass
            
    def _on_hover_leave(self):
        """Handle mouse leave with debouncing."""
        self._is_hovering = False
        # Delay the transition to prevent glitching
        if self._hover_job:
            self.after_cancel(self._hover_job)
        
        self._hover_job = self.after(self._transition_time, self._apply_normal_color)
        
    def _apply_normal_color(self):
        """Apply normal color after delay."""
        if not self._is_hovering:
            try:
                self.configure(fg_color=self._normal_color)
            except:
                pass
        self._hover_job = None


class StableButton(ctk.CTkButton, StableHoverMixin):
    """CTkButton with stable hover effects."""
    
    def __init__(self, *args, **kwargs):
        # Extract hover colors
        hover_color = kwargs.pop('hover_color', None)
        fg_color = kwargs.get('fg_color', '#1f538d')
        
        # Initialize button without hover_color to prevent default behavior
        super().__init__(*args, **kwargs)
        
        # Setup stable hover if hover color provided
        if hover_color:
            self.setup_stable_hover(fg_color, hover_color)


class StableFrame(ctk.CTkFrame):
    """CTkFrame with position stability to prevent shifting."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._locked_position = None
        
    def lock_position(self):
        """Lock the current position to prevent shifting."""
        self._locked_position = (self.winfo_x(), self.winfo_y())
        
    def maintain_position(self):
        """Maintain the locked position."""
        if self._locked_position:
            x, y = self._locked_position
            self.place(x=x, y=y)


def fix_hover_glitches(widget):
    """Apply hover fixes to a widget and its children."""
    # Fix buttons with hover effects
    if isinstance(widget, ctk.CTkButton):
        if hasattr(widget, '_hover_color'):
            # Disable the default hover animation
            widget._hover_animation_running = False
            
    # Recursively fix children
    for child in widget.winfo_children():
        fix_hover_glitches(child)


def stabilize_layout(container):
    """Stabilize layout to prevent shifting on hover."""
    # Set minimum sizes to prevent resizing
    if hasattr(container, 'pack_info'):
        info = container.pack_info()
        if 'ipadx' not in info:
            container.pack_configure(ipadx=5)
        if 'ipady' not in info:
            container.pack_configure(ipady=5)
    
    # Process all children
    for child in container.winfo_children():
        stabilize_layout(child)