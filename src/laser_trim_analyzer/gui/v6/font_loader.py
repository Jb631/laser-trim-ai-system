"""Load the bundled fonts (IBM Plex; Marcellus for titles): privately for Tk on Windows and macOS,
and for matplotlib everywhere.

Tk on Windows: CustomTkinter's FontManager.windows_load_font() calls AddFontResourceEx
PRIVATELY -- the font exists for this process only, no install, no admin, no IT request.
Its default passes FR_NOT_ENUM, which hides the family from tkinter.font.families(), and the
theme resolves families from that list -- so it would silently fall back to Segoe UI even with
Plex loaded. It is called with enumerable=True for exactly that reason.

macOS Tk (2026-10-04): each file is registered with CoreText for THIS process only
(CTFontManagerRegisterFontsForURL, scope "process") -- no install, gone when the app quits. Until
then the Mac never had Plex: Tk could not see it, and every Mac preview of the app was drawn in a
fallback system font. Linux Tk: nothing here; the theme's family tuples fall back to their next
entry. matplotlib reads TTF files directly, so charts get Plex on every platform.

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
         "IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf",
         # Titles: page titles, model names, the app's name (James picked it, 2026-10-04).
         "Marcellus-Regular.ttf")

# CoreText: register for this process only, and the code a second registration answers with.
_CT_SCOPE_PROCESS = 1
_CT_ERROR_ALREADY_REGISTERED = 105

_DONE: Optional[Dict[str, Dict[str, bool]]] = None


def _load_tk(path: Path) -> bool:
    """One clear warning line per failure -- not two. windows_load_font can fail two ways:
    it can return a falsy result (AddFontResourceEx just declined, no exception), or raise.
    Both are logged HERE, once, so load_bundled_fonts() doesn't also have to check and log
    the same failure a second time.
    """
    if sys.platform == "darwin":
        return _load_tk_mac(path)
    if not sys.platform.startswith("win"):
        return False
    try:
        from customtkinter import FontManager
        ok = bool(FontManager.windows_load_font(str(path), private=True, enumerable=True))
    except Exception:
        # A raised exception is less expected than a plain "no" -- log it as an error with its
        # traceback, as _load_matplotlib and the rest of the codebase do.
        logger.exception("bundled font %s did not load for the window; using the fallback",
                         path.name)
        return False
    if not ok:
        logger.warning("bundled font %s did not load for the window; using the fallback", path.name)
    return ok


def _load_tk_mac(path: Path) -> bool:
    """Register `path` with CoreText for this process only. A file this process already registered
    counts as loaded. Anything else that fails is logged once, and the fallback font is used."""
    try:
        import ctypes
        import ctypes.util
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        ct = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreText"))
        cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
        cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        cf.CFErrorGetCode.restype = ctypes.c_long
        cf.CFErrorGetCode.argtypes = [ctypes.c_void_p]
        ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
        ct.CTFontManagerRegisterFontsForURL.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        raw = str(path).encode()
        url = cf.CFURLCreateFromFileSystemRepresentation(None, raw, len(raw), False)
        if not url:
            logger.warning("bundled font %s could not be addressed; using the fallback", path.name)
            return False
        error = ctypes.c_void_p()
        try:
            ok = bool(ct.CTFontManagerRegisterFontsForURL(url, _CT_SCOPE_PROCESS,
                                                          ctypes.byref(error)))
        finally:
            cf.CFRelease(url)
        if not ok and error.value:
            code = cf.CFErrorGetCode(error)
            cf.CFRelease(error)
            if code == _CT_ERROR_ALREADY_REGISTERED:
                return True
        if not ok:
            logger.warning("bundled font %s did not load for the window; using the fallback",
                           path.name)
        return ok
    except Exception:
        logger.exception("bundled font %s did not load for the window; using the fallback",
                         path.name)
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
    if any(r["matplotlib"] for r in result.values()):
        import matplotlib
        matplotlib.rcParams["font.family"] = ["IBM Plex Sans", "DejaVu Sans"]
    _DONE = result
    return result
