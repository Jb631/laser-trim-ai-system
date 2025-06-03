"""
AlertBanner Widget for QA Dashboard

A modern, dismissible alert banner for displaying critical information.
Perfect for QA alerts, warnings, and notifications with smooth UI/UX.
"""

import tkinter as tk
from tkinter import ttk, font
from typing import Callable, Dict, List, Optional, Type
from datetime import datetime
import time


class AlertBanner(ttk.Frame):
    """
    Modern alert banner widget with smooth animations and proper state management.

    Features:
    - Multiple alert types (info, warning, error, success)
    - Smooth dismissal with CSS-like transitions
    - Action buttons with proper hover states
    - Auto-dismiss timer with cancellation
    - Icon support with visual feedback
    - Stack multiple alerts without UI conflicts
    - Scroll-aware positioning and management
    """

    def __init__(self, parent, alert_type: str = 'info',
                 title: str = "", message: str = "",
                 dismissible: bool = True,
                 auto_dismiss: Optional[int] = None,
                 actions: Optional[List[dict]] = None,
                 on_dismiss: Optional[Callable] = None,
                 allow_scroll: bool = True,
                 **kwargs):
        """
        Initialize AlertBanner with enhanced UI/UX features.

        Args:
            parent: Parent widget
            alert_type: Type of alert ('info', 'warning', 'error', 'success')
            title: Alert title
            message: Alert message
            dismissible: Whether alert can be dismissed
            auto_dismiss: Auto-dismiss after N seconds
            actions: List of action buttons [{'text': str, 'command': callable}]
            on_dismiss: Callback when alert is dismissed
            allow_scroll: Whether parent should remain scrollable with banner visible
        """
        super().__init__(parent, **kwargs)

        self.alert_type = alert_type
        self.title = title
        self.message = message
        self.dismissible = dismissible
        self.auto_dismiss = auto_dismiss
        self.actions = actions or []
        self.on_dismiss = on_dismiss
        self.allow_scroll = allow_scroll
        self.is_visible = True
        self.is_dismissing = False

        # Animation and timing controls
        self._animation_id = None
        self._auto_dismiss_id = None
        self._hover_animation_id = None
        self.animation_speed = 50  # milliseconds between animation frames
        self.fade_steps = 10  # number of steps for fade animation

        # Scroll management
        self._scroll_locked = False
        self._original_scroll_command = None

        # Colors and icons for different alert types
        self.alert_styles = {
            'info': {
                'bg': '#3498db',
                'fg': 'white',
                'icon': 'ℹ',
                'hover_bg': '#2980b9',
                'border': '#2980b9'
            },
            'warning': {
                'bg': '#f39c12',
                'fg': 'white',
                'icon': '⚠',
                'hover_bg': '#d68910',
                'border': '#d68910'
            },
            'error': {
                'bg': '#e74c3c',
                'fg': 'white',
                'icon': '✕',
                'hover_bg': '#c0392b',
                'border': '#c0392b'
            },
            'success': {
                'bg': '#27ae60',
                'fg': 'white',
                'icon': '✓',
                'hover_bg': '#229954',
                'border': '#229954'
            }
        }

        self.current_style = self.alert_styles.get(alert_type, self.alert_styles['info'])

        # Set up error boundary
        self._setup_error_boundary()
        
        # Setup UI with smooth appearance
        self._setup_ui_with_animation()

        # Start auto-dismiss timer if specified
        if self.auto_dismiss:
            self._auto_dismiss_id = self.after(self.auto_dismiss * 1000, self.dismiss)

    def _setup_error_boundary(self):
        """Set up comprehensive error handling for the banner."""
        def handle_banner_error(error_type, error_value, traceback):
            """Handle any banner-related errors gracefully."""
            try:
                print(f"AlertBanner Error: {error_type.__name__}: {error_value}")
                # Attempt graceful cleanup
                if self.is_visible:
                    self._emergency_cleanup()
            except:
                pass  # If even error handling fails, fail silently
                
        # Try to store original error handler safely
        try:
            # Get the root window for error reporting
            root = self.winfo_toplevel()
            if hasattr(root, 'report_callback_exception'):
                self._original_error_handler = root.report_callback_exception
                
                # Set up custom error handler for this banner
                def custom_error_handler(error_type, error_value, tb):
                    if "AlertBanner" in str(tb) or "alert_banner" in str(tb):
                        handle_banner_error(error_type, error_value, tb)
                    else:
                        # Pass through other errors to original handler
                        self._original_error_handler(error_type, error_value, tb)
                        
                # Set the custom error handler
                root.report_callback_exception = custom_error_handler
            else:
                # Fallback: just store the error handler function for cleanup
                self._original_error_handler = None
        except Exception as e:
            # If error boundary setup fails, continue without it
            print(f"Warning: Could not set up error boundary for AlertBanner: {e}")
            self._original_error_handler = None

    def _emergency_cleanup(self):
        """Emergency cleanup when banner errors occur."""
        try:
            # Cancel any running animations
            if self._animation_id:
                self.after_cancel(self._animation_id)
            if self._hover_animation_id:
                self.after_cancel(self._hover_animation_id)
            if self._auto_dismiss_id:
                self.after_cancel(self._auto_dismiss_id)
                
            # Restore scroll functionality
            self._restore_scroll()
            
            # Destroy widget
            self.destroy()
        except:
            pass  # Silent fail for emergency cleanup

    def _setup_ui_with_animation(self):
        """Set up the alert banner UI with smooth appearance animation."""
        # Configure frame style with smooth borders
        self.configure(relief='flat', borderwidth=1)
        
        # Create inner frame with enhanced styling
        self.inner_frame = tk.Frame(self, bg=self.current_style['bg'], 
                                   highlightbackground=self.current_style['border'],
                                   highlightthickness=1, bd=0)
        self.inner_frame.pack(fill='both', expand=True)

        # Manage scroll state
        if not self.allow_scroll:
            self._lock_scroll()

        # Create content frame with padding
        content_frame = tk.Frame(self.inner_frame, bg=self.current_style['bg'])
        content_frame.pack(fill='both', expand=True, padx=15, pady=12)

        # Left side - Icon and text
        left_frame = tk.Frame(content_frame, bg=self.current_style['bg'])
        left_frame.pack(side='left', fill='both', expand=True)

        # Icon with improved styling
        icon_label = tk.Label(left_frame, text=self.current_style['icon'],
                              font=('Segoe UI', 18, 'bold'),
                              bg=self.current_style['bg'],
                              fg=self.current_style['fg'])
        icon_label.pack(side='left', padx=(0, 12))

        # Text container
        text_frame = tk.Frame(left_frame, bg=self.current_style['bg'])
        text_frame.pack(side='left', fill='both', expand=True)

        # Title with improved typography
        if self.title:
            title_label = tk.Label(text_frame, text=self.title,
                                   font=('Segoe UI', 11, 'bold'),
                                   bg=self.current_style['bg'],
                                   fg=self.current_style['fg'],
                                   anchor='w')
            title_label.pack(anchor='w', pady=(0, 2))

        # Message with better text wrapping
        if self.message:
            message_label = tk.Label(text_frame, text=self.message,
                                     font=('Segoe UI', 10),
                                     bg=self.current_style['bg'],
                                     fg=self.current_style['fg'],
                                     anchor='w',
                                     wraplength=600,
                                     justify='left')
            message_label.pack(anchor='w')

        # Right side - Actions and dismiss
        right_frame = tk.Frame(content_frame, bg=self.current_style['bg'])
        right_frame.pack(side='right', padx=(15, 0))

        # Action buttons with enhanced styling
        self.action_buttons = []
        for action in self.actions:
            btn = tk.Button(right_frame, text=action['text'],
                            command=lambda cmd=action['command']: self._execute_action(cmd),
                            font=('Segoe UI', 9, 'bold'),
                            bg='white',
                            fg=self.current_style['bg'],
                            relief='flat',
                            padx=16, pady=6,
                            cursor='hand2',
                            bd=1,
                            highlightthickness=0)
            btn.pack(side='left', padx=(0, 8))
            self.action_buttons.append(btn)

            # Enhanced hover effects for buttons
            self._add_button_hover_effects(btn)

        # Dismiss button with smooth interactions
        if self.dismissible:
            self.dismiss_btn = tk.Label(right_frame, text='✕',
                                       font=('Segoe UI', 14, 'bold'),
                                       bg=self.current_style['bg'],
                                       fg=self.current_style['fg'],
                                       cursor='hand2',
                                       padx=8, pady=4)
            self.dismiss_btn.pack(side='right', padx=(12, 0))
            
            # Bind dismiss events with immediate response
            self.dismiss_btn.bind('<Button-1>', self._on_dismiss_click)
            self.dismiss_btn.bind('<Return>', self._on_dismiss_click)  # Keyboard accessibility
            
            # Enhanced hover effect for dismiss button
            self._add_dismiss_hover_effects()

        # Add overall banner hover effect
        self._add_banner_hover_effect()
        
        # Animate appearance
        self._animate_appearance()

    def _execute_action(self, command):
        """Execute action command with error boundary."""
        try:
            command()
        except Exception as e:
            print(f"Action command error: {e}")
            # Don't let action errors crash the banner

    def _add_button_hover_effects(self, button):
        """Add smooth hover effects to action buttons."""
        def on_enter(e):
            try:
                button.config(bg='#f8f9fa', relief='raised')
            except:
                pass

        def on_leave(e):
            try:
                button.config(bg='white', relief='flat')
            except:
                pass

        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)

    def _add_dismiss_hover_effects(self):
        """Add smooth hover effects to dismiss button."""
        def on_enter(e):
            try:
                self.dismiss_btn.config(font=('Segoe UI', 16, 'bold'), 
                                       bg=self.current_style['hover_bg'])
            except:
                pass

        def on_leave(e):
            try:
                self.dismiss_btn.config(font=('Segoe UI', 14, 'bold'),
                                       bg=self.current_style['bg'])
            except:
                pass

        self.dismiss_btn.bind('<Enter>', on_enter)
        self.dismiss_btn.bind('<Leave>', on_leave)

    def _add_banner_hover_effect(self):
        """Add subtle hover effect to entire banner."""
        def on_enter(e):
            try:
                self.inner_frame.config(bg=self.current_style['hover_bg'])
                self._update_bg_recursive(self.inner_frame, self.current_style['hover_bg'])
            except:
                pass

        def on_leave(e):
            try:
                self.inner_frame.config(bg=self.current_style['bg'])
                self._update_bg_recursive(self.inner_frame, self.current_style['bg'])
            except:
                pass

        self.inner_frame.bind('<Enter>', on_enter)
        self.inner_frame.bind('<Leave>', on_leave)

    def _update_bg_recursive(self, widget, bg_color):
        """Recursively update background color with error handling."""
        try:
            if widget.winfo_exists():
                widget.config(bg=bg_color)
                
                for child in widget.winfo_children():
                    if isinstance(child, (tk.Frame, tk.Label)) and \
                            not isinstance(child, tk.Button):
                        self._update_bg_recursive(child, bg_color)
        except (tk.TclError, AttributeError):
            pass  # Widget may have been destroyed

    def _animate_appearance(self):
        """Animate banner appearance with smooth slide-down effect."""
        # Start with height 0 and animate to full height
        self.configure(height=1)
        self._expand_banner(1, 70)  # Target height of 70px

    def _expand_banner(self, current_height: int, target_height: int):
        """Expand banner smoothly to target height."""
        if not self.is_visible or self.is_dismissing:
            return
            
        try:
            if current_height < target_height:
                # Smooth expansion
                next_height = min(target_height, current_height + 8)
                self.configure(height=next_height)
                self._animation_id = self.after(self.animation_speed, 
                                              lambda: self._expand_banner(next_height, target_height))
            else:
                # Animation complete - remove height constraint
                self.configure(height='')
                if self._animation_id:
                    self.after_cancel(self._animation_id)
                    self._animation_id = None
        except (tk.TclError, AttributeError):
            # Widget destroyed during animation
            pass

    def _lock_scroll(self):
        """Lock parent scroll functionality when banner requires attention."""
        try:
            parent = self.winfo_parent()
            if parent:
                parent_widget = self.nametowidget(parent)
                
                # Find scrollable parent
                current = parent_widget
                while current:
                    if hasattr(current, 'yview') and callable(current.yview):
                        # Found scrollable widget
                        self._original_scroll_command = getattr(current, 'yview', None)
                        # Disable scrolling by replacing yview
                        current.yview = lambda *args: None
                        self._scroll_locked = True
                        break
                    current = current.master if hasattr(current, 'master') else None
        except:
            pass  # If scroll lock fails, continue without it

    def _restore_scroll(self):
        """Restore parent scroll functionality."""
        try:
            if self._scroll_locked and self._original_scroll_command:
                parent = self.winfo_parent()
                if parent:
                    parent_widget = self.nametowidget(parent)
                    
                    # Find the same scrollable parent
                    current = parent_widget
                    while current:
                        if hasattr(current, 'yview'):
                            # Restore original scroll command
                            current.yview = self._original_scroll_command
                            self._scroll_locked = False
                            break
                        current = current.master if hasattr(current, 'master') else None
        except:
            pass  # If scroll restore fails, continue

    def _on_dismiss_click(self, event=None):
        """Handle dismiss button click with immediate response."""
        # Provide immediate visual feedback
        try:
            if hasattr(self, 'dismiss_btn'):
                self.dismiss_btn.config(font=('Segoe UI', 12, 'bold'))
                
            # Schedule actual dismissal slightly delayed for visual feedback
            self.after(50, self.dismiss)
        except:
            # If immediate feedback fails, just dismiss
            self.dismiss()

    def dismiss(self):
        """Dismiss the alert with smooth animation and proper cleanup."""
        if not self.is_visible or self.is_dismissing:
            return

        self.is_dismissing = True

        # Cancel auto-dismiss if active
        if self._auto_dismiss_id:
            try:
                self.after_cancel(self._auto_dismiss_id)
                self._auto_dismiss_id = None
            except:
                pass

        # Restore scroll immediately
        self._restore_scroll()

        # Execute callback before dismissal
        if self.on_dismiss:
            try:
                self.on_dismiss()
            except Exception as e:
                print(f"Dismiss callback error: {e}")

        # Use smooth fade-out animation
        self._animate_dismissal()

    def _animate_dismissal(self):
        """Animate banner dismissal with smooth fade and collapse."""
        try:
            # Get current height
            current_height = self.winfo_height()
            if current_height > 10:
                # Smooth collapse
                next_height = max(1, current_height - 12)  # Faster collapse
                self.configure(height=next_height)
                self._animation_id = self.after(self.animation_speed, self._animate_dismissal)
            else:
                # Animation complete - destroy widget
                self._final_cleanup()
        except (tk.TclError, AttributeError):
            # Widget already destroyed or error occurred
            self._final_cleanup()

    def _final_cleanup(self):
        """Final cleanup and widget destruction."""
        try:
            # Cancel any remaining animations
            if self._animation_id:
                self.after_cancel(self._animation_id)
            if self._hover_animation_id:
                self.after_cancel(self._hover_animation_id)
                
            # Ensure scroll is restored
            self._restore_scroll()
            
            # Destroy widget
            self.destroy()
        except:
            pass  # Silent cleanup

    def update_alert(self, title: Optional[str] = None,
                     message: Optional[str] = None,
                     alert_type: Optional[str] = None):
        """Update alert content with smooth transition."""
        try:
            if title is not None:
                self.title = title
            if message is not None:
                self.message = message
            if alert_type is not None and alert_type in self.alert_styles:
                self.alert_type = alert_type
                self.current_style = self.alert_styles[alert_type]

            # Smooth content update - fade out, update, fade in
            self._smooth_content_update()
        except Exception as e:
            print(f"Alert update error: {e}")

    def _smooth_content_update(self):
        """Update content with smooth transition."""
        try:
            # Rebuild UI
            for widget in self.winfo_children():
                widget.destroy()
            self._setup_ui_with_animation()
        except Exception as e:
            print(f"Content update error: {e}")


class AlertStack(ttk.Frame):
    """Container for stacking multiple alerts with proper UI management."""

    def __init__(self, parent, max_alerts: int = 5, spacing: int = 5, **kwargs):
        """
        Initialize AlertStack with enhanced UI management.

        Args:
            parent: Parent widget
            max_alerts: Maximum number of alerts to display
            spacing: Spacing between alerts
        """
        super().__init__(parent, **kwargs)

        self.max_alerts = max_alerts
        self.spacing = spacing
        self.alerts: List[AlertBanner] = []
        self._dismissing_alerts = set()  # Track alerts being dismissed

        # Configure style
        self.configure(relief='flat', borderwidth=0)

    def add_alert(self, alert_type: str = 'info',
                  title: str = "", message: str = "",
                  dismissible: bool = True,
                  auto_dismiss: Optional[int] = None,
                  actions: Optional[List[dict]] = None,
                  allow_scroll: bool = True) -> AlertBanner:
        """
        Add a new alert to the stack with proper management.

        Args:
            Same as AlertBanner plus allow_scroll

        Returns:
            The created AlertBanner instance
        """
        try:
            # Remove oldest alert if at max capacity
            if len(self.alerts) >= self.max_alerts:
                oldest_alert = self.alerts[0]
                if oldest_alert not in self._dismissing_alerts:
                    self._safe_dismiss_alert(oldest_alert)

            # Create new alert with error boundary
            alert = AlertBanner(
                self,
                alert_type=alert_type,
                title=title,
                message=message,
                dismissible=dismissible,
                auto_dismiss=auto_dismiss,
                actions=actions,
                allow_scroll=allow_scroll,
                on_dismiss=lambda: self._on_alert_dismiss(alert)
            )

            # Add to list and pack with proper spacing
            self.alerts.append(alert)
            alert.pack(fill='x', pady=(0, self.spacing))

            return alert
        except Exception as e:
            print(f"Error adding alert: {e}")
            # Return a dummy alert to prevent crashes
            return None

    def _safe_dismiss_alert(self, alert: AlertBanner):
        """Safely dismiss an alert with proper tracking."""
        try:
            if alert in self.alerts and alert not in self._dismissing_alerts:
                self._dismissing_alerts.add(alert)
                alert.dismiss()
        except Exception as e:
            print(f"Error dismissing alert: {e}")

    def _on_alert_dismiss(self, alert: AlertBanner):
        """Handle alert dismissal with proper cleanup."""
        try:
            if alert in self.alerts:
                self.alerts.remove(alert)
            if alert in self._dismissing_alerts:
                self._dismissing_alerts.remove(alert)
        except Exception as e:
            print(f"Error handling alert dismissal: {e}")

    def clear_all(self):
        """Clear all alerts with proper cleanup."""
        try:
            # Create copy to avoid modification during iteration
            alerts_to_clear = self.alerts.copy()
            self.alerts.clear()
            
            for alert in alerts_to_clear:
                if alert not in self._dismissing_alerts:
                    self._safe_dismiss_alert(alert)
                    
            # Clear dismissing set
            self._dismissing_alerts.clear()
        except Exception as e:
            print(f"Error clearing alerts: {e}")

    def get_alerts(self, alert_type: Optional[str] = None) -> List[AlertBanner]:
        """
        Get alerts by type with safe access.

        Args:
            alert_type: Filter by alert type (None for all)

        Returns:
            List of AlertBanner instances
        """
        try:
            if alert_type is None:
                return [a for a in self.alerts if a.is_visible]
            return [a for a in self.alerts if a.alert_type == alert_type and a.is_visible]
        except:
            return []


# Example usage and demonstration with enhanced UI/UX
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Enhanced AlertBanner Demo - QA Dashboard")
    root.geometry("900x700")
    root.configure(bg='#f5f5f5')

    # Title
    title_label = tk.Label(root, text="Enhanced QA Alert System Demo",
                           font=('Segoe UI', 16, 'bold'),
                           bg='#f5f5f5')
    title_label.pack(pady=20)

    # Create alert stack
    alert_stack = AlertStack(root, max_alerts=3)
    alert_stack.pack(fill='x', padx=20)

    # Demo functions with enhanced alerts
    def show_smooth_info():
        alert_stack.add_alert(
            alert_type='info',
            title='Enhanced Information',
            message='New analysis results with smooth UI interactions and proper scroll management.',
            actions=[
                {'text': 'View Results', 'command': lambda: print("Viewing results...")},
                {'text': 'Export', 'command': lambda: print("Exporting...")}
            ],
            allow_scroll=True
        )

    def show_scroll_lock_warning():
        alert_stack.add_alert(
            alert_type='warning',
            title='Critical Warning (Scroll Locked)',
            message='This warning locks scrolling until addressed. Notice smooth dismissal.',
            actions=[
                {'text': 'Address Issue', 'command': lambda: print("Addressing issue...")},
                {'text': 'Learn More', 'command': lambda: print("Learning more...")}
            ],
            allow_scroll=False,  # This will lock scrolling
            auto_dismiss=15
        )

    def show_smooth_error():
        alert_stack.add_alert(
            alert_type='error',
            title='Enhanced Error Handling',
            message='Error with improved UX and immediate visual feedback on interaction.',
            actions=[
                {'text': 'Retry', 'command': lambda: print("Retrying...")},
                {'text': 'Report Bug', 'command': lambda: print("Reporting bug...")}
            ]
        )

    def show_quick_success():
        alert_stack.add_alert(
            alert_type='success',
            title='Quick Success',
            message='Fast success notification with smooth animations.',
            auto_dismiss=3
        )

    # Control buttons
    button_frame = tk.Frame(root, bg='#f5f5f5')
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Smooth Info Alert", command=show_smooth_info,
              bg='#3498db', fg='white', padx=20, pady=10, relief='flat').pack(side='left', padx=5)

    tk.Button(button_frame, text="Scroll-Lock Warning", command=show_scroll_lock_warning,
              bg='#f39c12', fg='white', padx=20, pady=10, relief='flat').pack(side='left', padx=5)

    tk.Button(button_frame, text="Enhanced Error", command=show_smooth_error,
              bg='#e74c3c', fg='white', padx=20, pady=10, relief='flat').pack(side='left', padx=5)

    tk.Button(button_frame, text="Quick Success", command=show_quick_success,
              bg='#27ae60', fg='white', padx=20, pady=10, relief='flat').pack(side='left', padx=5)

    tk.Button(button_frame, text="Clear All Smoothly", command=alert_stack.clear_all,
              bg='#95a5a6', fg='white', padx=20, pady=10, relief='flat').pack(side='left', padx=5)

    # Add scrollable content to test scroll behavior
    content_frame = tk.Frame(root, bg='#f5f5f5')
    content_frame.pack(fill='both', expand=True, padx=20, pady=20)

    # Scrollable text area
    text_frame = tk.Frame(content_frame)
    text_frame.pack(fill='both', expand=True)

    canvas = tk.Canvas(text_frame, bg='white')
    scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg='white')

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Add content to test scrolling
    for i in range(50):
        tk.Label(scrollable_frame, text=f"Scrollable Content Line {i+1} - Test banner scroll behavior",
                font=('Segoe UI', 10), bg='white', anchor='w').pack(fill='x', pady=2, padx=10)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Info label
    info_label = tk.Label(root, 
                         text="Test smooth banner interactions, scroll behavior, and responsive dismissals",
                         font=('Segoe UI', 10, 'italic'),
                         bg='#f5f5f5', fg='#7f8c8d')
    info_label.pack(pady=10)

    root.mainloop()