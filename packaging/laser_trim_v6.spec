# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Laser Trim Analyzer (V6 UI) -- a Windows folder with LaserTrimAnalyzer.exe.
#
# Run it through scripts\build_exe.ps1 (which also checks the result). By hand, from the repo folder:
#     .venv\Scripts\python -m PyInstaller packaging\laser_trim_v6.spec --noconfirm --clean
#
# NOT RUN WHERE IT WAS WRITTEN. This file was written on a Mac with no PyInstaller installed
# (2026-09-30), so it is built to be right by construction and to say so when it is not:
#   * tests/test_packaging_files.py executes it with stand-ins for PyInstaller's names and checks
#     that every path it names exists and every hidden import it names can be imported;
#   * it uses only PyInstaller's oldest, plainest API (Analysis / PYZ / EXE / COLLECT with the
#     arguments below, SPECPATH, and collect_data_files). What is ASSUMED about PyInstaller, not
#     verified against a build, is marked "ASSUMED" where it is used;
#   * the build ends with `LaserTrimAnalyzer.exe --check`, which runs every part a packaged build
#     can be missing and names what is (laser_trim_analyzer/selfcheck.py).
#
# The choices, each for a reason:
#   ONEDIR, never onefile  -- a onefile .exe unpacks itself into TEMP at every start: slow, and
#                             exactly what an endpoint scanner distrusts. A folder starts at once.
#   console=False          -- she double-clicks it; no black window behind the app. Setting the
#                             environment variable LTA_CONSOLE=1 builds a console version instead
#                             (build_exe.ps1 -Console), where start-up errors are printed.
#   upx=False              -- UPX-compressed DLLs are a classic antivirus false positive.
#   no `data` folder       -- the build's output holds the app only. The database, config.yaml and
#                             the log live in a `data` folder BESIDE the .exe, on the computer that
#                             runs it, and an update must never touch that folder.
import json
import os
import subprocess
import sys

from PyInstaller.utils.hooks import collect_data_files

# ---- where things are -----------------------------------------------------------------------
# ASSUMED: PyInstaller defines SPECPATH (the folder holding this file) in a spec's namespace.
# Every path below is ABSOLUTE and built from it, so nothing depends on the folder the build is
# started from, nor on how PyInstaller resolves a relative path.
ROOT = os.path.dirname(os.path.abspath(SPECPATH))             # the repo folder
SRC = os.path.join(ROOT, "src")                               # the brief's pathex=["src"]
PACKAGE = os.path.join(SRC, "laser_trim_analyzer")
LAUNCHER = os.path.join(ROOT, "packaging", "laser_trim_v6_launcher.py")
FONTS = os.path.join(PACKAGE, "gui", "v6", "fonts")
for _needed in (SRC, PACKAGE, LAUNCHER, FONTS):
    if not os.path.exists(_needed):
        raise SystemExit("packaging/laser_trim_v6.spec: %s is not there -- is this spec still in "
                         "<repo>/packaging/?" % _needed)

CONSOLE = os.environ.get("LTA_CONSOLE", "") == "1"

# ---- the manifest: what this build must contain ----------------------------------------------
# The app's own self-check is run FROM SOURCE first, in this same Python (a child process, so
# none of the app's libraries are loaded into PyInstaller's own). It must pass: a build made from
# an environment that cannot run the app is not worth making. It then writes, as JSON:
#   modules -- every module of the app (walked from src/, not the in-package `tests`);
#   imports -- every module the app's own import statements name, top-level or inside a function;
#   loaded  -- every module that run actually LOADED from a file.
# `loaded` is handed to PyInstaller below as hidden imports, name by name. So nothing the
# self-check exercises -- pandas' Excel engines, matplotlib's backends, SQLAlchemy's dialect, the
# neighbours a compiled scikit-learn/scipy extension imports from C -- depends on PyInstaller's
# static analysis having seen it, and no library-version-specific module name is written here.
# The packaged app is then checked against `modules` and `imports` (it cannot list what it lacks).
# ASSUMED: PyInstaller defines `workpath` (its build folder); without it the repo's build/ is used.
_work = globals().get("workpath") or os.path.join(ROOT, "build", "laser_trim_v6")
MANIFEST = os.path.join(_work, "packaged_manifest.json")
_env = dict(os.environ)
_env["PYTHONPATH"] = SRC + os.pathsep + _env.get("PYTHONPATH", "")
_check = subprocess.run([sys.executable, "-B", "-m", "laser_trim_analyzer", "--check",
                         "--write-manifest", MANIFEST], cwd=ROOT, env=_env)
if _check.returncode != 0 or not os.path.isfile(MANIFEST):
    raise SystemExit("packaging/laser_trim_v6.spec: the app's self-check did not pass FROM SOURCE "
                     "in this environment (its lines are above). Fix that first -- no build was made.")
with open(MANIFEST, encoding="utf-8") as _f:
    _manifest = json.load(_f)

# ---- hidden imports ---------------------------------------------------------------------------
# Named here on purpose, whatever the manifest says -- each is found by NAME at run time, which
# no static analysis can follow:
HIDDEN_BY_NAME = [
    "xlrd",                                   # pandas' engine for .xls: every trim file
    "openpyxl",                               # pandas' engine for .xlsx: the exports, the backlog
    "matplotlib.backends.backend_tkagg",      # the chart canvas inside the window
    "matplotlib.backends.backend_agg",        # PNG export; drawing without a window
    "matplotlib.backends.backend_pdf",        # PDF export (savefig picks it from the file name)
    "matplotlib.backends.backend_svg",        # SVG export (the same)
    "sqlalchemy.dialects.sqlite",             # SQLAlchemy loads its dialect from the URL "sqlite://"
    "sqlalchemy.dialects.sqlite.pysqlite",
    "sqlite3",
    "yaml",                                   # data/config.yaml
    "PIL._tkinter_finder",                    # Pillow images inside Tk (CustomTkinter's CTkImage)
    "PIL.ImageTk",
]
HIDDENIMPORTS = sorted(set(HIDDEN_BY_NAME) | set(_manifest["modules"]) | set(_manifest["imports"])
                       | set(_manifest["loaded"]))

# ---- data files: (the file as it is now, the folder it lands in inside the build) -------------
# One tuple per FILE, so nothing rests on how a folder or a wildcard is copied.
DATAS = []
# The IBM Plex fonts: gui/v6/font_loader.py reads them from `fonts` beside itself -- the ONLY
# files the app reads from beside its own code (tests/test_packaged_mode.py pins that).
DATAS += [(os.path.join(FONTS, _name), "laser_trim_analyzer/gui/v6/fonts")
          for _name in sorted(os.listdir(FONTS)) if os.path.isfile(os.path.join(FONTS, _name))]
# The manifest, beside laser_trim_analyzer/selfcheck.py, which reads it in the packaged app.
DATAS += [(MANIFEST, "laser_trim_analyzer")]
# CustomTkinter's themes (JSON), its shapes font and its icon: it reads them from its own folder.
DATAS += collect_data_files("customtkinter")
# matplotlib's own data (mpl-data) and Tcl/Tk's libraries are NOT listed: ASSUMED to be collected
# by the hooks PyInstaller ships for matplotlib and tkinter. `--check` draws a chart and starts
# Tk, so a build without them says so.

# ---- what stays out ---------------------------------------------------------------------------
EXCLUDES = [
    "laser_trim_analyzer.tests",      # the in-package test stub
    "pytest", "_pytest",              # the test runner, if this venv has it
]

a = Analysis(
    [LAUNCHER],
    pathex=[SRC],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDENIMPORTS,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,            # ONEDIR: the binaries go to COLLECT below, not into the .exe
    name="LaserTrimAnalyzer",
    debug=False,
    strip=False,
    upx=False,
    console=CONSOLE,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LaserTrimAnalyzer",         # dist\LaserTrimAnalyzer\LaserTrimAnalyzer.exe
)
