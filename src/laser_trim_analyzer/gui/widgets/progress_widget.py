"""
ProgressWidget - Enhanced progress display with multiple styles

Provides various progress indication styles including circular,
linear, and step-based progress displays.
"""

import tkinter as tk
from tkinter import ttk
import math
from typing import Optional, List, Tuple
from datetime import datetime, timedelta


class ProgressWidget(ttk.Frame):
    """
    Multi-style progress widget for QA analysis.

    Features:
    - Linear progress bar
    - Circular progress indicator
    - Step-based progress
    - Time estimation
    - Pause/Cancel support
    """

    def __init__(
            self,
            parent,
            style: str = 'linear',
            show_percentage: bool = True,
            show_time_estimate: bool = True,
            steps: Optional[List[str]] = None,
            **kwargs
    ):
        """
        Initialize ProgressWidget.

        Args:
            parent: Parent widget
            style: Progress style ('linear', 'circular', 'steps')
            show_percentage: Show percentage text
            show_time_estimate: Show time remaining estimate
            steps: List of step names for step-based progress
        """
        super().__init__(parent, **kwargs)

        self.style_type = style
        self.show_percentage = show_percentage
        self.show_time_estimate = show_time_estimate
        self.steps = steps or []

        # Progress tracking
        self.progress = 0.0
        self.is_indeterminate = False
        self.is_paused = False
        self.start_time = None
        self.pause_time = None
        self.total_pause_duration = timedelta()

        # Colors
        self.colors = {
            'progress': '#2196f3',
            'background': '#e0e0e0',
            'success': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336',
            'text': '#212121',
            'text_light': '#757575'
        }

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI based on selected style."""
        if self.style_type == 'linear':
            self._setup_linear_progress()
        elif self.style_type == 'circular':
            self._setup_circular_progress()
        elif self.style_type == 'steps':
            self._setup_step_progress()
        else:
            self._setup_linear_progress()  # Default

    def _setup_linear_progress(self):
        """Set up linear progress bar style."""
        # Main container
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)

        # Top row: Label and percentage
        top_frame = ttk.Frame(container)
        top_frame.pack(fill='x', pady=(0, 5))

        self.label = ttk.Label(
            top_frame,
            text="Processing...",
            font=('Segoe UI', 10)
        )
        self.label.pack(side='left')

        if self.show_percentage:
            self.percentage_label = ttk.Label(
                top_frame,
                text="0%",
                font=('Segoe UI', 10, 'bold')
            )
            self.percentage_label.pack(side='right')

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            container,
            mode='determinate',
            maximum=100
        )
        self.progress_bar.pack(fill='x')

        # Bottom row: Status and time estimate
        bottom_frame = ttk.Frame(container)
        bottom_frame.pack(fill='x', pady=(5, 0))

        self.status_label = ttk.Label(
            bottom_frame,
            text="Ready",
            font=('Segoe UI', 9),
            foreground='gray'
        )
        self.status_label.pack(side='left')

        if self.show_time_estimate:
            self.time_label = ttk.Label(
                bottom_frame,
                text="",
                font=('Segoe UI', 9),
                foreground='gray'
            )
            self.time_label.pack(side='right')

    def _setup_circular_progress(self):
        """Set up circular progress indicator."""
        # Canvas for circular progress
        self.canvas = tk.Canvas(
            self,
            width=150,
            height=150,
            bg='white',
            highlightthickness=0
        )
        self.canvas.pack(pady=10)

        # Draw background circle
        self.canvas.create_oval(
            25, 25, 125, 125,
            outline=self.colors['background'],
            width=8,
            tags='background'
        )

        # Progress arc
        self.progress_arc = self.canvas.create_arc(
            25, 25, 125, 125,
            start=90,
            extent=0,
            outline=self.colors['progress'],
            width=8,
            style='arc',
            tags='progress'
        )

        # Center text
        self.center_text = self.canvas.create_text(
            75, 75,
            text="0%",
            font=('Segoe UI', 20, 'bold'),
            fill=self.colors['text']
        )

        # Labels below
        self.label = ttk.Label(
            self,
            text="Processing...",
            font=('Segoe UI', 10)
        )
        self.label.pack()

        self.status_label = ttk.Label(
            self,
            text="Ready",
            font=('Segoe UI', 9),
            foreground='gray'
        )
        self.status_label.pack()

    def _setup_step_progress(self):
        """Set up step-based progress display."""
        # Main container
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)

        # Overall progress
        top_frame = ttk.Frame(container)
        top_frame.pack(fill='x', pady=(0, 10))

        self.label = ttk.Label(
            top_frame,
            text="Processing...",
            font=('Segoe UI', 11, 'bold')
        )
        self.label.pack(side='left')

        self.percentage_label = ttk.Label(
            top_frame,
            text="0%",
            font=('Segoe UI', 11, 'bold')
        )
        self.percentage_label.pack(side='right')

        # Steps container
        self.steps_frame = ttk.Frame(container)
        self.steps_frame.pack(fill='both', expand=True)

        # Create step indicators
        self.step_widgets = []
        for i, step in enumerate(self.steps):
            step_widget = self._create_step_widget(self.steps_frame, step, i)
            step_widget.pack(fill='x', pady=2)
            self.step_widgets.append(step_widget)

        # Time estimate
        if self.show_time_estimate:
            self.time_label = ttk.Label(
                container,
                text="",
                font=('Segoe UI', 9),
                foreground='gray'
            )
            self.time_label.pack(pady=(10, 0))

    def _create_step_widget(self, parent, step_name: str, index: int) -> ttk.Frame:
        """Create a single step indicator widget."""
        frame = ttk.Frame(parent)

        # Step number circle
        number_canvas = tk.Canvas(
            frame,
            width=30,
            height=30,
            bg='white',
            highlightthickness=0
        )
        number_canvas.pack(side='left', padx=(0, 10))

        # Draw circle
        circle = number_canvas.create_oval(
            5, 5, 25, 25,
            fill=self.colors['background'],
            outline='',
            tags=f'circle_{index}'
        )

        # Step number
        number = number_canvas.create_text(
            15, 15,
            text=str(index + 1),
            font=('Segoe UI', 10, 'bold'),
            fill='white',
            tags=f'number_{index}'
        )

        # Store canvas reference
        frame.canvas = number_canvas
        frame.circle = circle
        frame.number = number

        # Step name
        name_label = ttk.Label(
            frame,
            text=step_name,
            font=('Segoe UI', 10)
        )
        name_label.pack(side='left', fill='x', expand=True)

        # Status icon
        status_label = ttk.Label(
            frame,
            text="",
            font=('Segoe UI', 12)
        )
        status_label.pack(side='right', padx=(10, 0))

        frame.name_label = name_label
        frame.status_label = status_label

        return frame

    def set_progress(self, value: float, status: Optional[str] = None):
        """
        Set progress value (0-100).

        Args:
            value: Progress percentage
            status: Optional status text
        """
        self.progress = max(0, min(100, value))
        self.is_indeterminate = False

        if self.start_time is None:
            self.start_time = datetime.now()

        # Update based on style
        if self.style_type == 'linear':
            self._update_linear_progress()
        elif self.style_type == 'circular':
            self._update_circular_progress()
        elif self.style_type == 'steps':
            self._update_step_progress()

        # Update status
        if status:
            self.set_status(status)

        # Update time estimate
        if self.show_time_estimate and self.progress > 0:
            self._update_time_estimate()

    def _update_linear_progress(self):
        """Update linear progress bar."""
        self.progress_bar['value'] = self.progress

        if self.show_percentage:
            self.percentage_label.config(text=f"{int(self.progress)}%")

        # Change color based on progress
        if self.progress >= 100:
            self.progress_bar.configure(style='Success.Horizontal.TProgressbar')

    def _update_circular_progress(self):
        """Update circular progress indicator."""
        # Calculate arc extent
        extent = -3.6 * self.progress  # Negative for clockwise

        self.canvas.itemconfig(
            self.progress_arc,
            extent=extent
        )

        # Update center text
        self.canvas.itemconfig(
            self.center_text,
            text=f"{int(self.progress)}%"
        )

        # Change color when complete
        if self.progress >= 100:
            self.canvas.itemconfig(
                self.progress_arc,
                outline=self.colors['success']
            )

    def _update_step_progress(self):
        """Update step-based progress."""
        if not self.steps:
            return

        # Calculate current step
        steps_count = len(self.steps)
        current_step = int(self.progress / 100 * steps_count)

        # Update step indicators
        for i, step_widget in enumerate(self.step_widgets):
            if i < current_step:
                # Completed step
                step_widget.canvas.itemconfig(
                    step_widget.circle,
                    fill=self.colors['success']
                )
                step_widget.status_label.config(text="✓")
            elif i == current_step:
                # Current step
                step_widget.canvas.itemconfig(
                    step_widget.circle,
                    fill=self.colors['progress']
                )
                step_widget.status_label.config(text="●")
            else:
                # Pending step
                step_widget.canvas.itemconfig(
                    step_widget.circle,
                    fill=self.colors['background']
                )
                step_widget.status_label.config(text="")

        # Update percentage
        self.percentage_label.config(text=f"{int(self.progress)}%")

    def _update_time_estimate(self):
        """Update time remaining estimate."""
        if not self.start_time or self.progress == 0:
            return

        # Calculate elapsed time (excluding pauses)
        now = datetime.now()
        elapsed = now - self.start_time - self.total_pause_duration

        # Estimate total time
        total_estimate = elapsed * (100 / self.progress)
        remaining = total_estimate - elapsed

        # Format time
        if remaining.total_seconds() < 60:
            time_text = f"{int(remaining.total_seconds())}s remaining"
        elif remaining.total_seconds() < 3600:
            minutes = int(remaining.total_seconds() / 60)
            time_text = f"{minutes}m remaining"
        else:
            hours = int(remaining.total_seconds() / 3600)
            minutes = int((remaining.total_seconds() % 3600) / 60)
            time_text = f"{hours}h {minutes}m remaining"

        if hasattr(self, 'time_label'):
            self.time_label.config(text=time_text)

    def set_indeterminate(self, start: bool = True):
        """Set progress to indeterminate mode."""
        self.is_indeterminate = start

        if self.style_type == 'linear' and hasattr(self, 'progress_bar'):
            if start:
                self.progress_bar.configure(mode='indeterminate')
                self.progress_bar.start(10)
                if self.show_percentage:
                    self.percentage_label.config(text="")
            else:
                self.progress_bar.stop()
                self.progress_bar.configure(mode='determinate')

    def set_status(self, status: str):
        """Update status text."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=status)

    def set_label(self, text: str):
        """Update main label text."""
        if hasattr(self, 'label'):
            self.label.config(text=text)

    def pause(self):
        """Pause progress tracking."""
        if not self.is_paused and self.start_time:
            self.is_paused = True
            self.pause_time = datetime.now()
            self.set_status("Paused")

    def resume(self):
        """Resume progress tracking."""
        if self.is_paused and self.pause_time:
            self.is_paused = False
            pause_duration = datetime.now() - self.pause_time
            self.total_pause_duration += pause_duration
            self.pause_time = None
            self.set_status("Resuming...")

    def reset(self):
        """Reset progress to zero."""
        self.progress = 0.0
        self.start_time = None
        self.pause_time = None
        self.total_pause_duration = timedelta()
        self.is_paused = False

        self.set_progress(0)
        self.set_status("Ready")

        if hasattr(self, 'time_label'):
            self.time_label.config(text="")

    def complete(self, message: str = "Complete"):
        """Mark progress as complete."""
        self.set_progress(100)
        self.set_status(message)

        if hasattr(self, 'time_label'):
            if self.start_time:
                elapsed = datetime.now() - self.start_time - self.total_pause_duration
                self.time_label.config(text=f"Completed in {elapsed.seconds}s")


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("ProgressWidget Demo")
    root.geometry("600x700")

    # Create notebook for different styles
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # Linear progress
    linear_frame = ttk.Frame(notebook)
    notebook.add(linear_frame, text="Linear")

    linear_progress = ProgressWidget(
        linear_frame,
        style='linear',
        show_percentage=True,
        show_time_estimate=True
    )
    linear_progress.pack(pady=20, padx=20, fill='x')

    # Circular progress
    circular_frame = ttk.Frame(notebook)
    notebook.add(circular_frame, text="Circular")

    circular_progress = ProgressWidget(
        circular_frame,
        style='circular'
    )
    circular_progress.pack(pady=20)

    # Step progress
    steps_frame = ttk.Frame(notebook)
    notebook.add(steps_frame, text="Steps")

    steps_progress = ProgressWidget(
        steps_frame,
        style='steps',
        steps=[
            "Loading files",
            "Extracting data",
            "Analyzing sigma",
            "Checking linearity",
            "Generating report"
        ]
    )
    steps_progress.pack(pady=20, padx=20, fill='both', expand=True)

    # Control buttons
    control_frame = ttk.Frame(root)
    control_frame.pack(pady=10)

    progress_var = tk.DoubleVar(value=0)


    def update_progress():
        value = progress_var.get()
        linear_progress.set_progress(value, f"Processing file {int(value)}...")
        circular_progress.set_progress(value)
        steps_progress.set_progress(value, f"Step {int(value / 20) + 1} of 5")

    def reset_all():
        linear_progress.reset()
        circular_progress.reset()
        steps_progress.reset()
        progress_var.set(0)

    def complete_all():
        linear_progress.complete()
        circular_progress.complete()
        steps_progress.complete()
        progress_var.set(100)

    scale = ttk.Scale(
        control_frame,
        from_=0,
        to=100,
        variable=progress_var,
        command=lambda x: update_progress(),
        length=300
    )
    scale.pack(pady=10)

    button_frame = ttk.Frame(control_frame)
    button_frame.pack()

    ttk.Button(
        button_frame,
        text="Reset",
        command=reset_all
    ).pack(side='left', padx=5)

    ttk.Button(
        button_frame,
        text="Complete",
        command=complete_all
    ).pack(side='left', padx=5)

    root.mainloop()