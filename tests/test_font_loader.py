"""The bundled fonts exist, are the families theme.py asks for, and fail loudly, not silently.

The Sans files come from IBM's own repository (Google's `ofl/ibmplexsans/` now carries only a
variable font) and the Mono files from Google's -- so there are two licence files, IBM's
`license.txt` (Sans) and Google's `OFL.txt` (Mono), both SIL OFL 1.1. Both are required.
"""
import logging
import sys
import types

from fontTools.ttLib import TTFont

from laser_trim_analyzer.gui.v6 import font_loader
from laser_trim_analyzer.gui.v6.theme import ThemeManager


def _family(path):
    """Name ID 1 -- the LEGACY family, which is what Windows GDI (and so Tk) uses."""
    return TTFont(str(path))["name"].getDebugName(1)


def test_every_bundled_file_and_both_licences_are_present():
    for name in font_loader.FILES:
        assert (font_loader.FONT_DIR / name).is_file(), name
    # OFL.txt (Google, covers the Mono files) and license.txt (IBM, covers the Sans files) --
    # neither one alone is the licence for everything bundled here.
    assert (font_loader.FONT_DIR / "OFL.txt").is_file()
    assert (font_loader.FONT_DIR / "license.txt").is_file()


def test_the_files_are_the_families_the_theme_asks_for():
    # This turns a Windows-only assumption into a fact checkable on any machine: GDI files
    # Plex Medium as its OWN family, which is why theme.font() maps "bold" onto it. Each
    # bundled file is checked against the specific tuple theme.py resolves it through --
    # not just whether SOME file matches SOME tuple's first entry -- so dropping the real
    # (abbreviated) spelling from FONT_FAMILY_MEDIUM turns this red even though the plain
    # "IBM Plex Sans Medium" spelling still sits in the tuple.
    t = ThemeManager
    owning_tuple = {
        "IBMPlexSans-Regular.ttf": t.FONT_FAMILY,
        "IBMPlexSans-Medium.ttf": t.FONT_FAMILY_MEDIUM,
        "IBMPlexMono-Regular.ttf": t.MONO_FAMILY,
        "IBMPlexMono-Medium.ttf": t.MONO_FAMILY_MEDIUM,
    }
    for name in font_loader.FILES:
        got = _family(font_loader.FONT_DIR / name)
        want = owning_tuple[name]
        assert got in want, f"{name} is family {got!r}, not in {want!r}"


def test_a_missing_file_falls_back_and_says_so(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(font_loader, "FONT_DIR", tmp_path)          # nothing in it
    monkeypatch.setattr(font_loader, "_DONE", None)
    with caplog.at_level(logging.WARNING):
        result = font_loader.load_bundled_fonts()
    assert all(not r["tk"] and not r["matplotlib"] for r in result.values())
    assert any("IBMPlexSans-Regular.ttf" in rec.getMessage() for rec in caplog.records)


def test_windows_tk_load_is_called_privately_and_enumerable(monkeypatch):
    """This Mac can never take the sys.platform.startswith("win") branch for real (Tk on
    macOS has no private-font call -- FR_PRIVATE is a Windows GDI flag), so fake the two
    things that differ: the platform check, and CustomTkinter's own windows_load_font. Pins
    the exact call: private=True (no install, no admin, no IT request) and -- the part that
    is easy to get wrong, because it is NOT windows_load_font's own default -- enumerable=True,
    or the family is hidden from tkinter.font.families() and the theme silently falls back to
    Segoe UI even with Plex loaded.
    """
    calls = []

    class FakeFontManager:
        @staticmethod
        def windows_load_font(path, private, enumerable):
            calls.append({"path": path, "private": private, "enumerable": enumerable})
            return True

    monkeypatch.setitem(sys.modules, "customtkinter", types.SimpleNamespace(FontManager=FakeFontManager))
    monkeypatch.setattr(sys, "platform", "win32")

    loaded = font_loader._load_tk(font_loader.FONT_DIR / "IBMPlexSans-Regular.ttf")

    assert loaded is True
    assert len(calls) == 1
    assert calls[0]["private"] is True
    assert calls[0]["enumerable"] is True
    assert calls[0]["path"] == str(font_loader.FONT_DIR / "IBMPlexSans-Regular.ttf")


def test_windows_tk_load_failure_is_logged_not_raised(monkeypatch, caplog):
    """Global constraint: a failed font load falls back and LOGS, never raises."""
    class ExplodingFontManager:
        @staticmethod
        def windows_load_font(path, private, enumerable):
            raise OSError("AddFontResourceEx refused (pretend locked file)")

    monkeypatch.setitem(sys.modules, "customtkinter", types.SimpleNamespace(FontManager=ExplodingFontManager))
    monkeypatch.setattr(sys, "platform", "win32")

    with caplog.at_level(logging.WARNING):
        loaded = font_loader._load_tk(font_loader.FONT_DIR / "IBMPlexSans-Regular.ttf")

    assert loaded is False
    assert any("IBMPlexSans-Regular.ttf" in rec.getMessage() for rec in caplog.records)


def test_matplotlib_can_use_plex_on_any_machine():
    import matplotlib.font_manager as fm
    font_loader.load_bundled_fonts()
    names = {f.name for f in fm.fontManager.ttflist}
    assert "IBM Plex Sans" in names and "IBM Plex Mono" in names


def test_loading_twice_is_harmless():
    a = font_loader.load_bundled_fonts()
    b = font_loader.load_bundled_fonts()
    assert a == b
