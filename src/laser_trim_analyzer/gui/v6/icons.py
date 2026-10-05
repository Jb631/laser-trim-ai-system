"""The app's icons: small white images (made from the Tabler icon set, MIT licence, by
scripts/build_icons.py) tinted to a theme colour and sized for CustomTkinter.

    icon("overview", theme.TEXT_SECONDARY, 18)  ->  a CTkImage, or None when the image is missing

None is never a crash: a widget given image=None shows its text alone, and the missing file is
logged once. One CTkImage per (name, colour, size), shared -- as theme.font() shares fonts.
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from PIL import Image

logger = logging.getLogger(__name__)

ICON_DIR = Path(__file__).resolve().parent / "icons"


def _rgb(color: str):
    h = color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@lru_cache(maxsize=None)
def _tinted(name: str, color: str) -> Optional[Image.Image]:
    path = ICON_DIR / f"{name}.png"
    try:
        mask = Image.open(path).convert("RGBA").getchannel("A")
    except (OSError, ValueError):
        logger.warning("icon %s is missing from %s; showing text only", name, ICON_DIR)
        return None
    img = Image.new("RGBA", mask.size, _rgb(color) + (255,))
    img.putalpha(mask)
    return img


@lru_cache(maxsize=None)
def icon(name: str, color: str, size: int = 18) -> Optional[ctk.CTkImage]:
    """`name` (a file in icons/, without .png) drawn in `color` at `size` CustomTkinter units."""
    img = _tinted(name, color)
    if img is None:
        return None
    return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
