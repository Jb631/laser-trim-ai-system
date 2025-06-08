"""
StatusBar Widget - Application status bar with connection indicators

Provides a modern status bar showing connection states, progress,
and system information.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Callable
from datetime import datetime
import threading
import time


class StatusIndicator(tk.Frame):
    """Individual status indicator with icon and label."""

    def __init__(self, parent, name: str, initial_status: str = "Unknown", **kwargs):
        super().__init__(parent, **kwargs)

        self.name = name
        self.status = initial_status

        # Status colors
        self.status_colors = {
            'connected': '#4caf50',
            'ready': '#4caf50',
            'disconnected': '#f44336',
            'error': '#f44336',
            'connecting': '#ff9800',
            'warning': '#ff9800',
            'unknown': '#9e9e9e',
            'disabled': '#9e9e9e'
        }

        self._setup_ui()

    def _setup_ui(self):
        """Set up the indicator UI."""
        # Icon (colored circle)
        self.icon_canvas = tk.Canvas(
            self,
            width=10,
            height=10,
            highlightthickness=0,
            bg=self.master.cget('bg')
        )
        self.icon_canvas.pack(side='left', padx=(0, 5))

        self.icon = self.icon_canvas.create_oval(
            2, 2, 8, 8,
            fill=self.status_colors.get('unknown', '#9e9e9e'),
            outline=''
        )

        # Label
        self.label = tk.Label(
            self,
            text=f"{self.name}: {self.status}",
            font=('Segoe UI', 9),
            bg=self.master.cget('bg'),
            fg='#424242'
        )
        self.label.pack(side='left')

    def update_status(self, status: str, animate: bool = True):
        """Update indicator status."""
        self.status = status
        self.label.configure(text=f"{self.name}: {status}")

        # Get color
        color_key = status.lower()
        color = self.status_colors.get(color_key, self.status_colors['unknown'])

        if animate and color_key in ['connecting', 'warning']:
            # Start pulsing animation
            self._start_pulse_animation(color)
        else:
            # Stop any animation and set solid color
            self._stop_animation()
            self.icon_canvas.itemconfig(self.icon, fill=color)

    def _start_pulse_animation(self, color: str):
        """Start pulsing animation for the indicator."""
        self._animating = True

        def pulse():
            fade_steps = 10
            while self._animating:
                # Fade out
                for i in range(fade_steps):
                    if not self._animating:
                        break
                    alpha = 1 - (i / fade_steps) * 0.5
                    faded_color = self._fade_color(color, alpha)
                    self.icon_canvas.itemconfig(self.icon, fill=faded_color)
                    time.sleep(0.05)

                # Fade in
                for i in range(fade_steps):
                    if not self._animating:
                        break
                    alpha = 0.5 + (i / fade_steps) * 0.5
                    faded_color = self._fade_color(color, alpha)
                    self.icon_canvas.itemconfig(self.icon, fill=faded_color)
                    time.sleep(0.05)

        thread = threading.Thread(target=pulse, daemon=True)
        thread.start()

    def _stop_animation(self):
        """Stop any running animation."""
        self._animating = False

    def _fade_color(self, color: str, alpha: float) -> str:
        """Fade a color by alpha amount."""
        # Convert hex to RGB
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        # Fade towards white
        r = int(r + (255 - r) * (1 - alpha))
        g = int(g + (255 - g) * (1 - alpha))
        b = int(b + (255 - b) * (1 - alpha))

        return f'#{r:02x}{g:02x}{b:02x}'


class StatusBar(tk.Frame):
    """
    Application status bar with multiple indicators.

    Features:
    - Connection status indicators
    - Progress display
    - Message area
    - System information
    - Click handlers for indicators
    """

    def __init__(
            self,
            parent,
            colors: Optional[Dict[str, str]] = None,
            height: int = 30,
            **kwargs
    ):
        """
        Initialize StatusBar.

        Args:
            parent: Parent widget
            colors: Color scheme dictionary
            height: Status bar height
        """
        super().__init__(parent, height=height, **kwargs)

        self.colors = colors or {
            'bg': '#f5f5f5',
            'border': '#e0e0e0',
            'text': '#424242',
            'text_secondary': '#757575'
        }

        self.indicators: Dict[str, StatusIndicator] = {}
        self.message_queue = []

        self._setup_ui()

    def _setup_ui(self):
        """Set up the status bar UI."""
        self.configure(bg=self.colors['bg'])

        # Top border
        border = tk.Frame(self, height=1, bg=self.colors['border'])
        border.pack(fill='x', side='top')

        # Main container
        container = tk.Frame(self, bg=self.colors['bg'])
        container.pack(fill='both', expand=True)

        # Left section - Status message
        left_frame = tk.Frame(container, bg=self.colors['bg'])
        left_frame.pack(side='left', fill='x', expand=True, padx=10)

        self.message_label = tk.Label(
            left_frame,
            text="Ready",
            font=('Segoe UI', 9),
            bg=self.colors['bg'],
            fg=self.colors['text'],
            anchor='w'
        )
        self.message_label.pack(fill='x')

        # Center section - Progress (optional)
        self.progress_frame = tk.Frame(container, bg=self.colors['bg'])
        # Don't pack initially

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='determinate',
            length=150
        )
        self.progress_bar.pack(pady=5)

        # Right section - Status indicators
        right_frame = tk.Frame(container, bg=self.colors['bg'])
        right_frame.pack(side='right', padx=10)

        # Default indicators container
        self.indicators_frame = tk.Frame(right_frame, bg=self.colors['bg'])
        self.indicators_frame.pack(side='left', padx=(0, 20))

        # Time display
        self.time_label = tk.Label(
            right_frame,
            text="",
            font=('Segoe UI', 9),
            bg=self.colors['bg'],
            fg=self.colors['text_secondary']
        )
        self.time_label.pack(side='right')

        # Start time update
        self._update_time()

    def add_indicator(
            self,
            name: str,
            label: str,
            initial_status: str = "Unknown",
            on_click: Optional[Callable] = None
    ) -> StatusIndicator:
        """
        Add a status indicator.

        Args:
            name: Unique indicator name
            label: Display label
            initial_status: Initial status
            on_click: Click callback

        Returns:
            Created StatusIndicator
        """
        # Create indicator
        indicator = StatusIndicator(
            self.indicators_frame,
            label,
            initial_status,
            bg=self.colors['bg']
        )
        indicator.pack(side='left', padx=(0, 15))

        # Store reference
        self.indicators[name] = indicator

        # Add click handler
        if on_click:
            indicator.bind("<Button-1>", lambda e: on_click())
            indicator.label.bind("<Button-1>", lambda e: on_click())
            indicator.icon_canvas.bind("<Button-1>", lambda e: on_click())

            # Change cursor on hover
            indicator.bind("<Enter>", lambda e: indicator.configure(cursor="hand2"))
            indicator.bind("<Leave>", lambda e: indicator.configure(cursor=""))

        return indicator

    def update_status(self, indicator_name: str, status: str):
        """Update a specific indicator status."""
        if indicator_name in self.indicators:
            self.indicators[indicator_name].update_status(status)

    def set_message(self, message: str, duration: Optional[int] = None):
        """
        Set status bar message.

        Args:
            message: Message text
            duration: Duration in milliseconds (None for permanent)
        """
        self.message_label.configure(text=message)

        if duration:
            # Auto-clear after duration
            self.after(duration, lambda: self.set_message("Ready"))

    def show_progress(self, value: Optional[float] = None, mode: str = 'determinate'):
        """
        Show progress bar.

        Args:
            value: Progress value (0-100) for determinate mode
            mode: 'determinate' or 'indeterminate'
        """
        self.progress_frame.pack(side='left', padx=20)

        if mode == 'indeterminate':
            self.progress_bar.configure(mode='indeterminate')
            self.progress_bar.start(10)
        else:
            self.progress_bar.configure(mode='determinate')
            if value is not None:
                self.progress_bar['value'] = value

    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()

    def push_message(self, message: str, duration: int = 3000):
        """
        Push a temporary message to the queue.

        Args:
            message: Message text
            duration: Display duration in milliseconds
        """
        self.message_queue.append((message, duration))

        if len(self.message_queue) == 1:
            self._process_message_queue()

    def _process_message_queue(self):
        """Process queued messages."""
        if not self.message_queue:
            self.set_message("Ready")
            return

        message, duration = self.message_queue.pop(0)
        self.set_message(message)

        # Schedule next message
        self.after(duration, self._process_message_queue)

    def _update_time(self):
        """Update time display."""
        current_time = datetime.now().strftime("%I:%M %p")
        self.time_label.configure(text=current_time)

        # Schedule next update
        self.after(1000, self._update_time)

    def create_default_indicators(self):
        """Create default status indicators."""
        # Database indicator
        self.add_indicator(
            'database',
            'DB',
            'Disconnected',
            on_click=lambda: self.push_message("Database connection details...", 2000)
        )

        # ML indicator
        self.add_indicator(
            'ml',
            'ML',
            'Not Available',
            on_click=lambda: self.push_message("ML model status...", 2000)
        )

        # API indicator
        self.add_indicator(
            'api',
            'API',
            'Not Connected',
            on_click=lambda: self.push_message("API connection details...", 2000)
        )


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("StatusBar Demo")
    root.geometry("800x600")

    # Main content
    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    ttk.Label(
        main_frame,
        text="Main Application Content",
        font=('Segoe UI', 24)
    ).pack(expand=True)

    # Status bar
    status_bar = StatusBar(root)
    status_bar.pack(fill='x', side='bottom')

    # Create default indicators
    status_bar.create_default_indicators()

    # Control buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)

    ttk.Button(
        button_frame,
        text="Connect DB",
        command=lambda: status_bar.update_status('database', 'Connected')
    ).pack(side='left', padx=5)

    ttk.Button(
        button_frame,
        text="ML Ready",
        command=lambda: status_bar.update_status('ml', 'Ready')
    ).pack(side='left', padx=5)

    ttk.Button(
        button_frame,
        text="API Connecting",
        command=lambda: status_bar.update_status('api', 'Connecting')
    ).pack(side='left', padx=5)

    # Progress control
    progress_frame = ttk.Frame(main_frame)
    progress_frame.pack(pady=20)

    progress_var = tk.DoubleVar()


    def update_progress(value):
        status_bar.show_progress(float(value))
        status_bar.set_message(f"Processing... {int(float(value))}%")


    scale = ttk.Scale(
        progress_frame,
        from_=0,
        to=100,
        variable=progress_var,
        command=update_progress,
        length=300
    )
    scale.pack()

    ttk.Button(
        progress_frame,
        text="Hide Progress",
        command=status_bar.hide_progress
    ).pack(pady=10)

    # Message controls
    msg_frame = ttk.Frame(main_frame)
    msg_frame.pack(pady=20)

    ttk.Button(
        msg_frame,
        text="Success Message",
        command=lambda: status_bar.push_message("✓ Operation completed successfully!", 3000)
    ).pack(side='left', padx=5)

    ttk.Button(
        msg_frame,
        text="Error Message",
        command=lambda: status_bar.push_message("✗ An error occurred!", 3000)
    ).pack(side='left', padx=5)

    ttk.Button(
        msg_frame,
        text="Multiple Messages",
        command=lambda: [
            status_bar.push_message("Processing file 1...", 1000),
            status_bar.push_message("Processing file 2...", 1000),
            status_bar.push_message("Processing file 3...", 1000),
            status_bar.push_message("All files processed!", 2000)
        ]
    ).pack(side='left', padx=5)

    root.mainloop()