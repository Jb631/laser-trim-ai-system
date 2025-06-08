"""
Theme Helper for Laser Trim Analyzer

Provides consistent theme colors and utilities for all widgets.
"""

import customtkinter as ctk
from typing import Dict, Tuple, Optional


class ThemeHelper:
    """Helper class for consistent theming across the application."""
    
    @staticmethod
    def get_theme_colors() -> Dict[str, Dict[str, str]]:
        """Get theme colors for current appearance mode."""
        mode = ctk.get_appearance_mode().lower()
        
        if mode == "dark":
            return {
                "bg": {"primary": "#212121", "secondary": "#2b2b2b", "tertiary": "#1f1f1f"},
                "fg": {"primary": "#ffffff", "secondary": "#e0e0e0", "tertiary": "#b0b0b0"},
                "accent": {"primary": "#1f538d", "secondary": "#144177", "tertiary": "#0e2e5a"},
                "border": {"primary": "#404040", "secondary": "#303030"},
                "hover": {"primary": "#333333", "secondary": "#404040"},
                "selection": {"primary": "#1f538d", "secondary": "#144177"},
                "success": "#4CAF50",
                "warning": "#FF9800",
                "error": "#F44336",
                "info": "#2196F3"
            }
        else:  # light mode
            return {
                "bg": {"primary": "#ffffff", "secondary": "#f5f5f5", "tertiary": "#e0e0e0"},
                "fg": {"primary": "#000000", "secondary": "#333333", "tertiary": "#666666"},
                "accent": {"primary": "#1976D2", "secondary": "#1565C0", "tertiary": "#0D47A1"},
                "border": {"primary": "#e0e0e0", "secondary": "#d0d0d0"},
                "hover": {"primary": "#f0f0f0", "secondary": "#e8e8e8"},
                "selection": {"primary": "#1976D2", "secondary": "#1565C0"},
                "success": "#4CAF50",
                "warning": "#FF9800",
                "error": "#F44336",
                "info": "#2196F3"
            }
    
    @staticmethod
    def get_ttk_style_config() -> Dict[str, any]:
        """Get TTK style configuration for current theme."""
        colors = ThemeHelper.get_theme_colors()
        
        return {
            "Treeview": {
                "configure": {
                    "background": colors["bg"]["secondary"],
                    "foreground": colors["fg"]["primary"],
                    "fieldbackground": colors["bg"]["secondary"],
                    "borderwidth": 0,
                    "lightcolor": colors["bg"]["secondary"],
                    "darkcolor": colors["bg"]["secondary"]
                },
                "map": {
                    "background": [('selected', colors["selection"]["primary"])],
                    "foreground": [('selected', colors["fg"]["primary"])]
                }
            },
            "Treeview.Heading": {
                "configure": {
                    "background": colors["bg"]["tertiary"],
                    "foreground": colors["fg"]["primary"],
                    "borderwidth": 1,
                    "relief": "flat",
                    "lightcolor": colors["bg"]["tertiary"],
                    "darkcolor": colors["bg"]["tertiary"]
                },
                "map": {
                    "background": [('active', colors["hover"]["primary"])]
                }
            },
            "TScrollbar": {
                "configure": {
                    "background": colors["bg"]["secondary"],
                    "troughcolor": colors["bg"]["primary"],
                    "bordercolor": colors["border"]["primary"],
                    "arrowcolor": colors["fg"]["secondary"],
                    "lightcolor": colors["bg"]["secondary"],
                    "darkcolor": colors["bg"]["secondary"]
                },
                "map": {
                    "background": [('active', colors["hover"]["primary"])]
                }
            }
        }
    
    @staticmethod
    def apply_ttk_style(style):
        """Apply theme to TTK style object."""
        config = ThemeHelper.get_ttk_style_config()
        
        for widget, settings in config.items():
            if "configure" in settings:
                style.configure(widget, **settings["configure"])
            if "map" in settings:
                style.map(widget, **settings["map"])
    
    @staticmethod
    def get_color(category: str, level: str = "primary") -> str:
        """Get a specific color from the theme."""
        colors = ThemeHelper.get_theme_colors()
        if category in colors:
            if isinstance(colors[category], dict) and level in colors[category]:
                return colors[category][level]
            elif isinstance(colors[category], str):
                return colors[category]
        return "#808080"  # fallback gray
    
    @staticmethod
    def get_frame_kwargs(level: str = "primary") -> Dict[str, any]:
        """Get kwargs for CTkFrame with proper theme colors."""
        colors = ThemeHelper.get_theme_colors()
        return {
            "fg_color": colors["bg"][level],
            "border_color": colors["border"]["primary"],
            "border_width": 1
        }
    
    @staticmethod
    def get_label_kwargs(level: str = "primary") -> Dict[str, any]:
        """Get kwargs for CTkLabel with proper theme colors."""
        colors = ThemeHelper.get_theme_colors()
        return {
            "text_color": colors["fg"][level]
        }
    
    @staticmethod
    def get_text_kwargs() -> Dict[str, any]:
        """Get kwargs for CTkTextbox with proper theme colors."""
        colors = ThemeHelper.get_theme_colors()
        return {
            "fg_color": colors["bg"]["secondary"],
            "text_color": colors["fg"]["primary"],
            "border_color": colors["border"]["primary"],
            "border_width": 1
        }
    
    @staticmethod
    def get_transparent_frame_kwargs() -> Dict[str, any]:
        """Get kwargs for transparent CTkFrame that works with themes."""
        # Using 'transparent' can cause issues, so we use the parent's bg color
        return {
            "fg_color": "transparent",
            "corner_radius": 0
        }