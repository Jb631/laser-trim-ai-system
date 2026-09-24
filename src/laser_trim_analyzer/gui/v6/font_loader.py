"""Load the bundled IBM Plex fonts: privately for Tk on Windows, and for matplotlib everywhere.

Tk on Windows: CustomTkinter's FontManager.windows_load_font() calls AddFontResourceEx
PRIVATELY -- the font exists for this process only, no install, no admin, no IT request.
Its default passes FR_NOT_ENUM, which hides the family from tkinter.font.families(), and the
theme resolves families from that list -- so it would silently fall back to Segoe UI even with
Plex loaded. It is called with enumerable=True for exactly that reason.

macOS/Linux Tk: nothing here (FontManager.windows_load_font uses ctypes.windll, which only
exists on Windows); the theme's family tuples fall back to their next entry. matplotlib reads
TTF files directly, so charts get Plex on every platform, including this one.

A file that fails to load is logged at WARNING and the app continues on the fallback fonts --
never silently, never fatally.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FILES = ("IBMPlexSans-Regular.ttf", "IBMPlexSans-Medium.ttf",
         "IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf")

_DONE: Optional[Dict[str, Dict[str, bool]]] = None


def _load_tk(path: Path) -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        from customtkinter import FontManager
        return bool(FontManager.windows_load_font(str(path), private=True, enumerable=True))
    except Exception:
        logger.exception("could not load %s for the window", path.name)
        return False


def _load_matplotlib(path: Path) -> bool:
    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(str(path))
        return True
    except Exception:
        logger.exception("could not load %s for charts", path.name)
        return False


def load_bundled_fonts() -> Dict[str, Dict[str, bool]]:
    """Load every bundled file once. Returns {file: {"tk": bool, "matplotlib": bool}}."""
    global _DONE
    if _DONE is not None:
        return _DONE
    result: Dict[str, Dict[str, bool]] = {}
    for name in FILES:
        path = FONT_DIR / name
        if not path.is_file():
            logger.warning("bundled font %s is missing from %s; using the fallback fonts",
                           name, FONT_DIR)
            result[name] = {"tk": False, "matplotlib": False}
            continue
        result[name] = {"tk": _load_tk(path), "matplotlib": _load_matplotlib(path)}
        if sys.platform.startswith("win") and not result[name]["tk"]:
            logger.warning("bundled font %s did not load for the window; using the fallback", name)
    if any(r["matplotlib"] for r in result.values()):
        import matplotlib
        matplotlib.rcParams["font.family"] = ["IBM Plex Sans", "DejaVu Sans"]
    _DONE = result
    return result
