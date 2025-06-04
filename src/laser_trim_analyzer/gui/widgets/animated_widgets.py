"""
Animated UI Widgets for Enhanced User Experience

Provides smooth animations, transitions, and accessibility features
for the Laser Trim Analyzer application.
"""

import tkinter as tk
import customtkinter as ctk
from typing import Callable, Optional, Dict, Any
import threading
import time
import math


class AnimatedProgressBar(ctk.CTkProgressBar):
    """Progress bar with smooth animation."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._target_value = 0
        self._current_value = 0
        self._animation_speed = 0.02
        self._animation_running = False
    
    def animate_to(self, value: float, duration: float = 1.0):
        """Animate to target value over specified duration."""
        self._target_value = max(0, min(1, value))
        self._animation_speed = abs(self._target_value - self._current_value) / (duration * 60)  # 60 FPS
        
        if not self._animation_running:
            self._start_animation()
    
    def _start_animation(self):
        """Start the animation loop."""
        self._animation_running = True
        self._animate_step()
    
    def _animate_step(self):
        """Single animation step."""
        if abs(self._target_value - self._current_value) < 0.001:
            self._current_value = self._target_value
            self._animation_running = False
            self.set(self._current_value)
            return
        
        # Smooth easing function
        diff = self._target_value - self._current_value
        self._current_value += diff * 0.1
        
        self.set(self._current_value)
        
        if self._animation_running:
            self.after(16, self._animate_step)  # ~60 FPS


class FadeInFrame(ctk.CTkFrame):
    """Frame with fade-in animation."""
    
    def __init__(self, parent, fade_duration=0.5, **kwargs):
        super().__init__(parent, **kwargs)
        self.fade_duration = fade_duration
        self._alpha = 0.0
        self._target_alpha = 1.0
        self._fade_step = 0.05
        
        # Initially transparent
        self.configure(fg_color=self._get_alpha_color(self.cget("fg_color"), 0))
    
    def fade_in(self):
        """Start fade-in animation."""
        self._alpha = 0.0
        self._target_alpha = 1.0
        self._animate_fade()
    
    def fade_out(self, callback: Optional[Callable] = None):
        """Start fade-out animation."""
        self._alpha = 1.0
        self._target_alpha = 0.0
        self._fade_callback = callback
        self._animate_fade()
    
    def _animate_fade(self):
        """Animate fade effect."""
        if abs(self._alpha - self._target_alpha) < 0.01:
            self._alpha = self._target_alpha
            if hasattr(self, '_fade_callback') and self._fade_callback:
                self._fade_callback()
            return
        
        diff = self._target_alpha - self._alpha
        self._alpha += diff * 0.15
        
        # Update appearance based on alpha
        color = self._get_alpha_color(self.cget("fg_color"), self._alpha)
        self.configure(fg_color=color)
        
        self.after(16, self._animate_fade)
    
    def _get_alpha_color(self, base_color, alpha):
        """Get color with alpha blending."""
        # Simplified alpha blending - in practice you'd want more sophisticated color handling
        try:
            if isinstance(base_color, str) and base_color.startswith('#'):
                # Extract RGB values
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)
                
                # Blend with white background
                r = int(r * alpha + 255 * (1 - alpha))
                g = int(g * alpha + 255 * (1 - alpha))
                b = int(b * alpha + 255 * (1 - alpha))
                
                return f"#{r:02x}{g:02x}{b:02x}"
        except:
            pass
        
        return base_color


class SlideInFrame(ctk.CTkFrame):
    """Frame with slide-in animation."""
    
    def __init__(self, parent, slide_direction='right', slide_distance=300, **kwargs):
        super().__init__(parent, **kwargs)
        self.slide_direction = slide_direction
        self.slide_distance = slide_distance
        self._original_x = 0
        self._original_y = 0
        self._target_x = 0
        self._target_y = 0
        self._animating = False
    
    def slide_in(self):
        """Start slide-in animation."""
        if self._animating:
            return
            
        # Store original position
        self._original_x = self.winfo_x()
        self._original_y = self.winfo_y()
        
        # Set initial position based on direction
        if self.slide_direction == 'right':
            start_x = self._original_x + self.slide_distance
            start_y = self._original_y
        elif self.slide_direction == 'left':
            start_x = self._original_x - self.slide_distance
            start_y = self._original_y
        elif self.slide_direction == 'down':
            start_x = self._original_x
            start_y = self._original_y + self.slide_distance
        else:  # up
            start_x = self._original_x
            start_y = self._original_y - self.slide_distance
        
        self.place(x=start_x, y=start_y)
        
        self._target_x = self._original_x
        self._target_y = self._original_y
        self._current_x = start_x
        self._current_y = start_y
        
        self._animating = True
        self._animate_slide()
    
    def _animate_slide(self):
        """Animate slide movement."""
        if not self._animating:
            return
            
        # Check if animation is complete
        if (abs(self._current_x - self._target_x) < 1 and 
            abs(self._current_y - self._target_y) < 1):
            self.place(x=self._target_x, y=self._target_y)
            self._animating = False
            return
        
        # Smooth easing
        diff_x = self._target_x - self._current_x
        diff_y = self._target_y - self._current_y
        
        self._current_x += diff_x * 0.15
        self._current_y += diff_y * 0.15
        
        self.place(x=int(self._current_x), y=int(self._current_y))
        
        self.after(16, self._animate_slide)


class AnimatedButton(ctk.CTkButton):
    """Button with hover and click animations."""
    
    def __init__(self, parent, **kwargs):
        # Store original colors
        self._original_fg_color = kwargs.get('fg_color', ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        self._original_hover_color = kwargs.get('hover_color', ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        
        super().__init__(parent, **kwargs)
        
        # Bind hover events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        
        self._scale = 1.0
        self._animating_scale = False
    
    def _on_enter(self, event):
        """Handle mouse enter with animation."""
        if not self._animating_scale:
            self._animate_scale(1.05, duration=0.1)
    
    def _on_leave(self, event):
        """Handle mouse leave with animation."""
        if not self._animating_scale:
            self._animate_scale(1.0, duration=0.1)
    
    def _on_press(self, event):
        """Handle button press with animation."""
        self._animate_scale(0.95, duration=0.05)
    
    def _on_release(self, event):
        """Handle button release with animation."""
        self._animate_scale(1.05, duration=0.05)
    
    def _animate_scale(self, target_scale: float, duration: float = 0.2):
        """Animate button scale."""
        self._target_scale = target_scale
        self._scale_speed = abs(target_scale - self._scale) / (duration * 60)
        
        if not self._animating_scale:
            self._animating_scale = True
            self._scale_step()
    
    def _scale_step(self):
        """Single scale animation step."""
        if abs(self._scale - self._target_scale) < 0.01:
            self._scale = self._target_scale
            self._animating_scale = False
            return
        
        diff = self._target_scale - self._scale
        self._scale += diff * 0.3
        
        # Apply visual scaling effect (simplified)
        self.after(16, self._scale_step)


class LoadingSpinner(ctk.CTkFrame):
    """Animated loading spinner."""
    
    def __init__(self, parent, size=50, color="#1f538d", **kwargs):
        super().__init__(parent, width=size, height=size, **kwargs)
        
        self.size = size
        self.color = color
        self._rotation = 0
        self._spinning = False
        
        # Create canvas for drawing
        self.canvas = tk.Canvas(
            self,
            width=size,
            height=size,
            bg=self.cget("fg_color")[1],
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)
        
        self._draw_spinner()
    
    def start_spinning(self):
        """Start spinner animation."""
        if not self._spinning:
            self._spinning = True
            self._spin_step()
    
    def stop_spinning(self):
        """Stop spinner animation."""
        self._spinning = False
    
    def _spin_step(self):
        """Single spin animation step."""
        if not self._spinning:
            return
            
        self._rotation += 10
        if self._rotation >= 360:
            self._rotation = 0
        
        self._draw_spinner()
        self.after(50, self._spin_step)  # 20 FPS for smooth rotation
    
    def _draw_spinner(self):
        """Draw the spinner at current rotation."""
        self.canvas.delete("all")
        
        center = self.size // 2
        radius = self.size // 4
        
        # Draw spinning arcs
        for i in range(8):
            angle = (i * 45 + self._rotation) % 360
            alpha = 1.0 - (i / 8.0)  # Fade effect
            
            # Calculate arc position
            start_angle = angle
            extent = 30
            
            # Draw arc with varying opacity (simplified)
            arc_color = self._blend_color(self.color, "#ffffff", 1 - alpha)
            
            self.canvas.create_arc(
                center - radius, center - radius,
                center + radius, center + radius,
                start=start_angle, extent=extent,
                outline=arc_color, width=3,
                style='arc'
            )
    
    def _blend_color(self, color1, color2, ratio):
        """Blend two colors by ratio."""
        # Simplified color blending
        try:
            if color1.startswith('#') and color2.startswith('#'):
                r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
                r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
                
                r = int(r1 * ratio + r2 * (1 - ratio))
                g = int(g1 * ratio + g2 * (1 - ratio))
                b = int(b1 * ratio + b2 * (1 - ratio))
                
                return f"#{r:02x}{g:02x}{b:02x}"
        except:
            pass
        
        return color1


class AnimatedNotification(ctk.CTkToplevel):
    """Animated notification popup."""
    
    def __init__(self, parent, message: str, notification_type: str = "info", duration: float = 3.0):
        super().__init__(parent)
        
        self.message = message
        self.notification_type = notification_type
        self.duration = duration
        
        # Configure window
        self.withdraw()  # Hide initially
        self.transient(parent)
        self.overrideredirect(True)  # Remove window decorations
        
        # Color schemes
        colors = {
            "info": {"bg": "#2196F3", "text": "white"},
            "success": {"bg": "#4CAF50", "text": "white"},
            "warning": {"bg": "#FF9800", "text": "white"},
            "error": {"bg": "#F44336", "text": "white"}
        }
        
        color_scheme = colors.get(notification_type, colors["info"])
        
        # Create content
        self.configure(fg_color=color_scheme["bg"])
        
        self.label = ctk.CTkLabel(
            self,
            text=message,
            text_color=color_scheme["text"],
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.label.pack(padx=20, pady=15)
        
        # Position notification
        self._position_notification()
        
        # Show with animation
        self.show_animated()
    
    def _position_notification(self):
        """Position notification in top-right corner."""
        self.update_idletasks()
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        
        x = screen_width - width - 20
        y = 20
        
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def show_animated(self):
        """Show notification with slide-in animation."""
        self.deiconify()
        
        # Start from off-screen
        original_x = self.winfo_x()
        start_x = original_x + self.winfo_width()
        
        self.geometry(f"+{start_x}+{self.winfo_y()}")
        
        # Animate slide-in
        self._animate_slide_in(start_x, original_x)
        
        # Schedule auto-hide
        self.after(int(self.duration * 1000), self.hide_animated)
    
    def _animate_slide_in(self, start_x: int, target_x: int):
        """Animate slide-in effect."""
        current_x = start_x
        
        def slide_step():
            nonlocal current_x
            
            if abs(current_x - target_x) < 1:
                self.geometry(f"+{target_x}+{self.winfo_y()}")
                return
            
            diff = target_x - current_x
            current_x += diff * 0.2
            
            self.geometry(f"+{int(current_x)}+{self.winfo_y()}")
            self.after(16, slide_step)
        
        slide_step()
    
    def hide_animated(self):
        """Hide notification with slide-out animation."""
        target_x = self.winfo_x() + self.winfo_width()
        current_x = self.winfo_x()
        
        def slide_out_step():
            nonlocal current_x
            
            if current_x >= target_x:
                self.destroy()
                return
            
            diff = target_x - current_x
            current_x += diff * 0.2
            
            self.geometry(f"+{int(current_x)}+{self.winfo_y()}")
            self.after(16, slide_out_step)
        
        slide_out_step()


class AccessibilityHelper:
    """Helper class for accessibility features."""
    
    @staticmethod
    def add_keyboard_navigation(widget, on_enter_callback: Callable = None):
        """Add keyboard navigation support to widget."""
        def on_key(event):
            if event.keysym == 'Return' or event.keysym == 'space':
                if on_enter_callback:
                    on_enter_callback()
                elif hasattr(widget, 'invoke'):
                    widget.invoke()
        
        widget.bind('<Key>', on_key)
        widget.focus_set()  # Make focusable
    
    @staticmethod
    def add_tooltips(widget, tooltip_text: str):
        """Add accessible tooltips to widget."""
        tooltip = None
        
        def show_tooltip(event):
            nonlocal tooltip
            x, y, cx, cy = widget.bbox("insert") if hasattr(widget, 'bbox') else (0, 0, 0, 0)
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            
            label = tk.Label(
                tooltip,
                text=tooltip_text,
                background="lightyellow",
                relief="solid",
                borderwidth=1,
                font=("Arial", 10, "normal")
            )
            label.pack()
        
        def hide_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None
        
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
    
    @staticmethod
    def set_high_contrast_mode(root, enabled: bool = True):
        """Enable high contrast mode for better accessibility."""
        if enabled:
            # High contrast color scheme
            ctk.set_appearance_mode("dark")
            
            # You could also modify individual widget colors here
            # This is a simplified example
        else:
            ctk.set_appearance_mode("light")


def show_notification(parent, message: str, notification_type: str = "info", duration: float = 3.0):
    """Convenience function to show animated notification."""
    return AnimatedNotification(parent, message, notification_type, duration) 