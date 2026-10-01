"""What scripts/build_exe.ps1 needs that is better said in Python -- because Python can be tested
on the machine the code is written on, and PowerShell cannot (tests/test_packaging_files.py).

    python packaging\\build_support.py check-pyinstaller
        exit 0: PyInstaller is installed in this Python and supports it
        exit 3: PyInstaller is not installed
        exit 4: the installed PyInstaller does not support this Python (says both versions)

    python packaging\\build_support.py stamp dist\\LaserTrimAnalyzer
        writes build_info.txt (which commit, when, with what) and READ ME FIRST.txt into that
        folder, and prints the stamp. git missing, or not a checkout: "unknown", never a failure.

Standard library only. It reads the stamp's FORMAT from the app (laser_trim_analyzer.selfcheck),
so the file the build writes is the file the app reads.
"""
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laser_trim_analyzer.selfcheck import (BUILD_INFO_NAME, describe_build,  # noqa: E402
                                           format_build_info)

README_NAME = "READ ME FIRST.txt"

# Plain ASCII, on purpose: it is opened in Notepad on any Windows code page.
README = """\
LASER TRIM ANALYZER
===================

STARTING IT

1. Unzip this folder anywhere on your own computer (the Desktop or Documents is fine).
   Do not put it in OneDrive or on a network drive.

2. The app only ever opens the "data" folder that sits beside LaserTrimAnalyzer.exe,
   in this folder, on this computer. It holds your own copy of the data (analysis.db
   and config.yaml). Without it the app starts empty.

3. Double-click LaserTrimAnalyzer.exe. The first start takes a moment.


GOOD TO KNOW

- Everything you look at comes from the copy in the "data" folder here. Nothing you
  do in this app changes anyone else's data.

- The "Predictor" panel says there is no predictor: trained predictors are not
  loaded on this computer. Nothing you view needs them.

- If the app does not start, or shows an error, send the file data\\laser_trim.log
  from this folder to the person who gave you the app.


UPDATING TO A NEWER VERSION

1. Close the app.

2. In this folder, delete everything EXCEPT the "data" folder.

3. Unzip the new version, and move everything that is inside its LaserTrimAnalyzer
   folder into this folder, so that LaserTrimAnalyzer.exe sits beside "data" again.

4. Start LaserTrimAnalyzer.exe. The first start after an update can take longer:
   the app upgrades the database by itself.


This build: {build}
"""


def _git(*args: str) -> str:
    """One line from git about this checkout, or "" when it cannot be had."""
    try:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=30)
    except Exception:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _version_of(distribution: str) -> str:
    try:
        from importlib.metadata import version
        return version(distribution)
    except Exception:
        return ""


def build_info(now: "datetime.datetime | None" = None) -> dict:
    """The stamp of a build made now, from this checkout: `unknown` where it cannot be told.
    A commit built while what the build is MADE from (src/, packaging/, the pinned
    requirements) has uncommitted changes says so (`abc1234+changes`): a stamp that named the
    commit alone would name code the build does not contain."""
    commit = _git("rev-parse", "--short", "HEAD")
    if commit and _git("status", "--porcelain", "--", "src", "packaging",
                       "requirements-pinned.txt"):
        commit += "+changes"
    now = now or datetime.datetime.now()
    return {
        "commit": commit,
        "commit_date": _git("log", "-1", "--format=%cd", "--date=format:%Y-%m-%d"),
        "built": now.strftime("%Y-%m-%d %H:%M"),
        "python": "%d.%d.%d" % sys.version_info[:3],
        "pyinstaller": _version_of("pyinstaller"),
    }


def stamp(dist_dir, now=None) -> dict:
    """Write build_info.txt and READ ME FIRST.txt into the built folder; returns the stamp."""
    dist_dir = Path(dist_dir)
    info = build_info(now)
    with open(dist_dir / BUILD_INFO_NAME, "w", encoding="ascii", errors="replace",
              newline="\r\n") as f:
        f.write(format_build_info(info))
    with open(dist_dir / README_NAME, "w", encoding="ascii", errors="replace",
              newline="\r\n") as f:
        f.write(README.format(build=describe_build(info)))
    return info


# ---- does the installed PyInstaller support this Python? --------------------------------------

def _version_tuple(text: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", text)[:3])


def python_satisfies(requires_python: str, version: tuple) -> bool:
    """Is `version` (e.g. (3, 12, 7)) inside a Requires-Python specifier such as
    ">=3.8,<3.15"? A small reader of exactly the clauses such a line holds; anything it does
    not understand counts as satisfied (the build itself is then the judge)."""
    for clause in (c.strip() for c in (requires_python or "").split(",")):
        m = re.fullmatch(r"(>=|<=|==|!=|~=|<|>)\s*([0-9][0-9.]*)(\.\*)?", clause)
        if not m:
            continue
        op, want = m.group(1), _version_tuple(m.group(2))
        have = version[:len(want)] if (m.group(3) or op in ("==", "!=")) else version
        padded = want + (0,) * (len(have) - len(want))
        if op == ">=" and not have >= padded:
            return False
        if op == "<=" and not have <= padded:
            return False
        if op == "<" and not have < padded:
            return False
        if op == ">" and not have > padded:
            return False
        if op == "==" and have[:len(want)] != want:
            return False
        if op == "!=" and have[:len(want)] == want:
            return False
        if op == "~=" and not (have >= padded and have[:len(want) - 1] == want[:-1]):
            return False
    return True


def check_pyinstaller(out=print) -> int:
    this = "%d.%d.%d" % sys.version_info[:3]
    try:
        from importlib.metadata import PackageNotFoundError, metadata
        try:
            meta = metadata("pyinstaller")
        except PackageNotFoundError:
            out(f"PyInstaller is not installed in this Python ({this}, {sys.executable}).")
            return 3
    except Exception as e:                       # pragma: no cover - a broken environment
        out(f"Could not ask this Python about PyInstaller ({type(e).__name__}: {e}).")
        return 3
    version, requires = meta.get("Version", "?"), meta.get("Requires-Python", "") or ""
    if not python_satisfies(requires, sys.version_info[:3]):
        out(f"PyInstaller {version} does not support this Python: it needs Python {requires}, "
            f"and .venv has {this}.")
        out("Either upgrade it (  .venv\\Scripts\\python -m pip install --upgrade pyinstaller  "
            "-- this downloads from PyPI), or rebuild .venv with a Python it supports.")
        return 4
    out(f"PyInstaller {version} supports this Python ({this}; it needs {requires or 'any'}).")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check-pyinstaller"] and len(argv) == 1:
        return check_pyinstaller()
    if argv[:1] == ["stamp"] and len(argv) == 2:
        dist_dir = Path(argv[1])
        if not dist_dir.is_dir():
            print(f"no such folder: {dist_dir}")
            return 2
        info = stamp(dist_dir)
        print(f"build {describe_build(info)}")
        print(f"wrote {BUILD_INFO_NAME} and {README_NAME} into {dist_dir}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
