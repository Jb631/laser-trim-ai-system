"""
AlertBanner Widget for QA Dashboard

A modern, dismissible alert banner for displaying critical information.
Perfect for QA alerts, warnings, and notifications.
"""

import tkinter as tk
from tkinter import ttk, font
from typing import Callable, Dict, List, Optional, Typefrom datetime import datetime
import time


class AlertBanner(ttk.Frame):
    """
    Modern alert banner widget with animations and actions.

    Features:
    - Multiple alert types (info, warning, error, success)
    - Dismissible with animation
    - Action buttons
    - Auto-dismiss timer
    - Icon support
    - Stack multiple alerts
    """

    def __init__(self, parent, alert_type: str = 'info',
                 title: str = "", message: str = "",
                 dismissible: bool = True,
                 auto_dismiss: Optional[int] = None,
                 actions: Optional[List[dict]] = None,
                 on_dismiss: Optional[Callable] = None,
                 **kwargs):
        """
        Initialize AlertBanner.

        Args:
            parent: Parent widget
            alert_type: Type of alert ('info', 'warning', 'error', 'success')
            title: Alert title
            message: Alert message
            dismissible: Whether alert can be dismissed
            auto_dismiss: Auto-dismiss after N seconds
            actions: List of action buttons [{'text': str, 'command': callable}]
            on_dismiss: Callback when alert is dismissed
        """
        super().__init__(parent, **kwargs)

        self.alert_type = alert_type
        self.title = title
        self.message = message
        self.dismissible = dismissible
        self.auto_dismiss = auto_dismiss
        self.actions = actions or []
        self.on_dismiss = on_dismiss
        self.is_visible = True

        # Colors and icons for different alert types
        self.alert_styles = {
            'info': {
                'bg': '#3498db',
                'fg': 'white',
                'icon': 'ℹ',
                'hover_bg': '#2980b9'
            },
            'warning': {
                'bg': '#f39c12',
                'fg': 'white',
                'icon': '⚠',
                'hover_bg': '#d68910'
            },
            'error': {
                'bg': '#e74c3c',
                'fg': 'white',
                'icon': '✕',
                'hover_bg': '#c0392b'
            },
            'success': {
                'bg': '#27ae60',
                'fg': 'white',
                'icon': '✓',
                'hover_bg': '#229954'
            }
        }

        self.current_style = self.alert_styles.get(alert_type, self.alert_styles['info'])
        self._animation_id = None
        self._auto_dismiss_id = None

        self._setup_ui()

        # Start auto-dismiss timer if specified
        if self.auto_dismiss:
            self._auto_dismiss_id = self.after(self.auto_dismiss * 1000, self.dismiss)

    def _setup_ui(self):
        """Set up the alert banner UI."""
        # Configure frame style
        self.configure(relief='flat', borderwidth=0)

        # Create inner frame with padding
        self.inner_frame = tk.Frame(self, bg=self.current_style['bg'])
        self.inner_frame.pack(fill='both', expand=True)

        # Create content frame
        content_frame = tk.Frame(self.inner_frame, bg=self.current_style['bg'])
        content_frame.pack(fill='both', expand=True, padx=15, pady=10)

        # Left side - Icon and text
        left_frame = tk.Frame(content_frame, bg=self.current_style['bg'])
        left_frame.pack(side='left', fill='both', expand=True)

        # Icon
        icon_label = tk.Label(left_frame, text=self.current_style['icon'],
                              font=('Segoe UI', 20, 'bold'),
                              bg=self.current_style['bg'],
                              fg=self.current_style['fg'])
        icon_label.pack(side='left', padx=(0, 10))

        # Text container
        text_frame = tk.Frame(left_frame, bg=self.current_style['bg'])
        text_frame.pack(side='left', fill='both', expand=True)

        # Title
        if self.title:
            title_label = tk.Label(text_frame, text=self.title,
                                   font=('Segoe UI', 11, 'bold'),
                                   bg=self.current_style['bg'],
                                   fg=self.current_style['fg'],
                                   anchor='w')
            title_label.pack(anchor='w')

        # Message
        if self.message:
            message_label = tk.Label(text_frame, text=self.message,
                                     font=('Segoe UI', 10),
                                     bg=self.current_style['bg'],
                                     fg=self.current_style['fg'],
                                     anchor='w',
                                     wraplength=500)
            message_label.pack(anchor='w')

        # Right side - Actions and dismiss
        right_frame = tk.Frame(content_frame, bg=self.current_style['bg'])
        right_frame.pack(side='right', padx=(10, 0))

        # Action buttons
        for action in self.actions:
            btn = tk.Button(right_frame, text=action['text'],
                            command=action['command'],
                            font=('Segoe UI', 9, 'bold'),
                            bg='white',
                            fg=self.current_style['bg'],
                            relief='flat',
                            padx=15, pady=5,
                            cursor='hand2')
            btn.pack(side='left', padx=(0, 5))

            # Hover effect
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg='#ecf0f1'))
            btn.bind('<Leave>', lambda e, b=btn: b.config(bg='white'))

        # Dismiss button
        if self.dismissible:
            dismiss_btn = tk.Label(right_frame, text='✕',
                                   font=('Segoe UI', 12),
                                   bg=self.current_style['bg'],
                                   fg=self.current_style['fg'],
                                   cursor='hand2')
            dismiss_btn.pack(side='right', padx=(10, 0))
            dismiss_btn.bind('<Button-1>', lambda e: self.dismiss())

            # Hover effect
            dismiss_btn.bind('<Enter>',
                             lambda e: dismiss_btn.config(font=('Segoe UI', 14)))
            dismiss_btn.bind('<Leave>',
                             lambda e: dismiss_btn.config(font=('Segoe UI', 12)))

        # Add hover effect to entire banner
        self._add_hover_effect(self.inner_frame)

    def _add_hover_effect(self, widget):
        """Add hover effect to widget and its children."""

        def on_enter(e):
            self.inner_frame.config(bg=self.current_style['hover_bg'])
            self._update_bg_recursive(self.inner_frame, self.current_style['hover_bg'])

        def on_leave(e):
            self.inner_frame.config(bg=self.current_style['bg'])
            self._update_bg_recursive(self.inner_frame, self.current_style['bg'])

        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)

    def _update_bg_recursive(self, widget, bg_color):
        """Recursively update background color."""
        try:
            widget.config(bg=bg_color)
        except:
            pass

        for child in widget.winfo_children():
            if isinstance(child, (tk.Frame, tk.Label)) and \
                    not isinstance(child, tk.Button):
                self._update_bg_recursive(child, bg_color)

    def dismiss(self):
        """Dismiss the alert with animation."""
        if not self.is_visible:
            return

        self.is_visible = False

        # Cancel auto-dismiss if active
        if self._auto_dismiss_id:
            self.after_cancel(self._auto_dismiss_id)

        self._animate_slide_up()

        if self.on_dismiss:
            self.on_dismiss()

    def _animate_slide_up(self):
        """Animate sliding up to dismiss."""
        current_height = self.winfo_height()
        if current_height > 5:
            # Reduce height
            self.configure(height=current_height - 5)
            self._animation_id = self.after(10, self._animate_slide_up)
        else:
            # Animation complete, destroy widget
            if self._animation_id:
                self.after_cancel(self._animation_id)
            self.destroy()

    def update_alert(self, title: Optional[str] = None,
                     message: Optional[str] = None,
                     alert_type: Optional[str] = None):
        """Update alert content."""
        if title is not None:
            self.title = title
        if message is not None:
            self.message = message
        if alert_type is not None and alert_type in self.alert_styles:
            self.alert_type = alert_type
            self.current_style = self.alert_styles[alert_type]

        # Rebuild UI
        for widget in self.winfo_children():
            widget.destroy()
        self._setup_ui()


class AlertStack(ttk.Frame):
    """Container for stacking multiple alerts."""

    def __init__(self, parent, max_alerts: int = 5, spacing: int = 5, **kwargs):
        """
        Initialize AlertStack.

        Args:
            parent: Parent widget
            max_alerts: Maximum number of alerts to display
            spacing: Spacing between alerts
        """
        super().__init__(parent, **kwargs)

        self.max_alerts = max_alerts
        self.spacing = spacing
        self.alerts: List[AlertBanner] = []

        # Configure style
        self.configure(relief='flat', borderwidth=0)

    def add_alert(self, alert_type: str = 'info',
                  title: str = "", message: str = "",
                  dismissible: bool = True,
                  auto_dismiss: Optional[int] = None,
                  actions: Optional[List[dict]] = None) -> AlertBanner:
        """
        Add a new alert to the stack.

        Args:
            Same as AlertBanner

        Returns:
            The created AlertBanner instance
        """
        # Remove oldest alert if at max capacity
        if len(self.alerts) >= self.max_alerts:
            oldest_alert = self.alerts.pop(0)
            oldest_alert.dismiss()

        # Create new alert
        alert = AlertBanner(
            self,
            alert_type=alert_type,
            title=title,
            message=message,
            dismissible=dismissible,
            auto_dismiss=auto_dismiss,
            actions=actions,
            on_dismiss=lambda: self._on_alert_dismiss(alert)
        )

        # Add to list and pack
        self.alerts.append(alert)
        alert.pack(fill='x', pady=(0, self.spacing))

        # Animate entrance
        self._animate_slide_down(alert)

        return alert

    def _on_alert_dismiss(self, alert: AlertBanner):
        """Handle alert dismissal."""
        if alert in self.alerts:
            self.alerts.remove(alert)

    def _animate_slide_down(self, alert: AlertBanner):
        """Animate alert sliding down."""
        alert.configure(height=1)
        self._expand_alert(alert, 1)

    def _expand_alert(self, alert: AlertBanner, current_height: int):
        """Expand alert to full height."""
        if current_height < 60:  # Target height
            alert.configure(height=current_height + 3)
            self.after(10, lambda: self._expand_alert(alert, current_height + 3))
        else:
            alert.configure(height='')  # Reset to natural height

    def clear_all(self):
        """Clear all alerts."""
        for alert in self.alerts[:]:  # Copy list to avoid modification during iteration
            alert.dismiss()
        self.alerts.clear()

    def get_alerts(self, alert_type: Optional[str] = None) -> List[AlertBanner]:
        """
        Get alerts by type.

        Args:
            alert_type: Filter by alert type (None for all)

        Returns:
            List of AlertBanner instances
        """
        if alert_type is None:
            return self.alerts.copy()
        return [a for a in self.alerts if a.alert_type == alert_type]


# Example usage and demonstration
if __name__ == "__main__":
    root = tk.Tk()
    root.title("AlertBanner Demo - QA Dashboard")
    root.geometry("800x600")
    root.configure(bg='#f5f5f5')

    # Title
    title_label = tk.Label(root, text="QA Alert System Demo",
                           font=('Segoe UI', 16, 'bold'),
                           bg='#f5f5f5')
    title_label.pack(pady=20)

    # Create alert stack
    alert_stack = AlertStack(root)
    alert_stack.pack(fill='x', padx=20)


    # Demo functions
    def show_info_alert():
        alert_stack.add_alert(
            alert_type='info',
            title='Information',
            message='5 new analysis results are ready for review.',
            actions=[
                {'text': 'View Results', 'command': lambda: print("Viewing results...")},
                {'text': 'Export', 'command': lambda: print("Exporting...")}
            ]
        )


    def show_warning_alert():
        alert_stack.add_alert(
            alert_type='warning',
            title='Sigma Threshold Warning',
            message='3 units have sigma gradients approaching threshold limits.',
            auto_dismiss=10,
            actions=[
                {'text': 'Review Units', 'command': lambda: print("Reviewing units...")}
            ]
        )


    def show_error_alert():
        alert_stack.add_alert(
            alert_type='error',
            title='Critical Failure',
            message='Unit 8340-A12345 has exceeded all tolerance limits and requires immediate attention.',
            dismissible=False,
            actions=[
                {'text': 'View Details', 'command': lambda: print("Viewing details...")},
                {'text': 'Notify Team', 'command': lambda: print("Notifying team...")}
            ]
        )


    def show_success_alert():
        alert_stack.add_alert(
            alert_type='success',
            title='Analysis Complete',
            message='All 50 units passed quality checks. Reports have been generated.',
            auto_dismiss=5
        )


    # Control buttons
    button_frame = tk.Frame(root, bg='#f5f5f5')
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Show Info Alert", command=show_info_alert,
              bg='#3498db', fg='white', padx=20, pady=10).pack(side='left', padx=5)

    tk.Button(button_frame, text="Show Warning", command=show_warning_alert,
              bg='#f39c12', fg='white', padx=20, pady=10).pack(side='left', padx=5)

    tk.Button(button_frame, text="Show Error", command=show_error_alert,
              bg='#e74c3c', fg='white', padx=20, pady=10).pack(side='left', padx=5)

    tk.Button(button_frame, text="Show Success", command=show_success_alert,
              bg='#27ae60', fg='white', padx=20, pady=10).pack(side='left', padx=5)

    tk.Button(button_frame, text="Clear All", command=alert_stack.clear_all,
              bg='#95a5a6', fg='white', padx=20, pady=10).pack(side='left', padx=5)

    # Info label
    info_label = tk.Label(root, text="Click buttons to show different alert types",
                          font=('Segoe UI', 10, 'italic'),
                          bg='#f5f5f5', fg='#7f8c8d')
    info_label.pack(pady=10)

    root.mainloop()