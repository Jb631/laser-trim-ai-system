"""`--check`: the app checks itself, without a window -- and the stamp of a packaged build.

    LaserTrimAnalyzer.exe --check            the packaged app (scripts/build_exe.ps1 runs this)
    python -m laser_trim_analyzer --check    from source

Why it exists (2026-09-30). The app is also handed out as a PyInstaller folder, built on the work
laptop and never on the machine the code is written on. A packaged build can be INCOMPLETE in
ways the source never is: a module PyInstaller's static analysis did not see (pandas picks its
Excel engine by name, matplotlib its backend, SQLAlchemy its dialect, a compiled extension imports
its neighbours from C), or a data file nobody listed (the fonts). Each of those fails LATE -- on
the page, the export or the file that first needs it. This runs every one of them up front, by
name, in a few seconds, and says which:

    one line per step, then  PACKAGED BUILD OK
                         or  PACKAGED BUILD INCOMPLETE: <the steps that failed>      exit 0 / 1

It works from source too (that is how it is tested), where the same final line means "this
environment can run everything the app does".

What it never does: open a window you can see, open the app's own database (`data` is only
LOOKED at -- which database the app WOULD open is reported, because that is the question a copy
handed to someone else raises), or write anything beside the app except `check_result.txt` in a
packaged build (a windowed build has no console to print to).

A packaged build is checked against a MANIFEST written when it was built
(`packaging/laser_trim_v6.spec` runs this file's own check from source first, with
`--write-manifest`): the app's modules and every module its import statements name. The list
cannot come from the packaged app itself -- a module that was left out is not there to be listed.

Also here: the build stamp. `scripts/build_exe.ps1` writes `build_info.txt` beside the .exe;
`read_build_info` reads it (never raising), `--check` prints it and the app logs it once at start.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import io
import json
import logging
import os
import re
import struct
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from laser_trim_analyzer import config as _config

PACKAGE = "laser_trim_analyzer"

OK, NOTE, FAIL = "ok", "note", "FAIL"
OK_LINE = "PACKAGED BUILD OK"
INCOMPLETE_LINE = "PACKAGED BUILD INCOMPLETE"

BUILD_INFO_NAME = "build_info.txt"          # beside the .exe; written by the build script
CHECK_RESULT_NAME = "check_result.txt"      # beside the .exe; written by --check when packaged
MANIFEST_NAME = "packaged_manifest.json"    # inside the bundle, beside this module

# What build_info.txt holds, one `key: value` line each, in this order.
BUILD_INFO_KEYS = ("commit", "commit_date", "built", "python", "pyinstaller")

# Loaded while the check runs from source, but never the packaged app's business: the build's own
# machinery, the test runner, and what a venv's .pth files load at start-up.
_NOT_THE_APPS = ("PyInstaller", "pytest", "_pytest", "_distutils_hack", "pip", "sitecustomize",
                 "usercustomize")


class CheckFailed(Exception):
    """A step's own verdict: its message is the whole explanation (no traceback is added)."""


def is_frozen() -> bool:
    """True in a packaged build (PyInstaller sets sys.frozen)."""
    return bool(getattr(sys, "frozen", False))


def app_directory() -> Path:
    """The app's folder, asked through the config module every time (the tests' seam)."""
    return Path(_config.get_app_directory())


# =============================================================================================
# the build stamp

def format_build_info(info: Dict[str, str]) -> str:
    """The text of build_info.txt: every key, `unknown` where the build could not tell."""
    return "".join(f"{key}: {info.get(key) or 'unknown'}\n" for key in BUILD_INFO_KEYS)


def read_build_info(app_dir: Union[str, Path, None] = None) -> Optional[Dict[str, str]]:
    """The build stamp beside the app, or None when there is none (running from source).

    Never raises. A file that is there but cannot be read, or holds nothing usable, is `{}` --
    "a packaged build that cannot say which" -- so the caller still knows a stamp was expected.
    PowerShell 5.1 writes UTF-16 with some commands and ANSI with others: both are read.
    """
    path = Path(app_dir if app_dir is not None else app_directory()) / BUILD_INFO_NAME
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read(4096)
    except OSError:
        return {}
    try:
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            text = raw.decode("utf-16", errors="replace")
        else:
            text = raw.decode("utf-8-sig", errors="replace")
    except Exception:
        return {}
    info: Dict[str, str] = {}
    for line in text.splitlines():
        key, colon, value = line.partition(":")
        key = key.strip().lower()
        if not colon or not re.fullmatch(r"[a-z_]{1,32}", key):
            continue
        value = "".join(ch for ch in " ".join(value.split()) if ch.isprintable())[:120]
        if value:
            info.setdefault(key, value)
    return info


def build_line(info: Optional[Dict[str, str]]) -> Optional[str]:
    """`build <commit> of <its date>` -- what the app logs once at start; None without a stamp."""
    if info is None:
        return None
    return f"build {info.get('commit') or 'unknown'} of {info.get('commit_date') or 'unknown'}"


def describe_build(info: Dict[str, str]) -> str:
    """The whole stamp on one line, for `--check`."""
    return (f"{info.get('commit') or 'unknown'} of {info.get('commit_date') or 'unknown'}; "
            f"built {info.get('built') or 'unknown'} with Python "
            f"{info.get('python') or 'unknown'} and PyInstaller "
            f"{info.get('pyinstaller') or 'unknown'}")


# =============================================================================================
# what a build must contain

def package_directory() -> Path:
    """Where this package is: the source tree, or the bundle in a packaged build."""
    return Path(__file__).resolve().parent


def app_modules(package_dir: Union[str, Path, None] = None) -> List[str]:
    """Every module of the app by dotted name, walked from a SOURCE tree: not the in-package
    `tests`, and not `__main__` (the entry point itself -- importing it again would set logging
    up a second time)."""
    root = Path(package_dir) if package_dir is not None else package_directory()
    names = []
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[0] == "tests" or parts[-1] == "__main__":
            continue
        if parts[-1] == "__init__":
            parts.pop()
        names.append(".".join([PACKAGE] + parts))
    return names


def named_imports(package_dir: Union[str, Path, None] = None) -> List[str]:
    """Every name the app's own import statements could mean as a module, outside the app
    itself: `import a.b` gives a.b; `from a import b` gives a and a.b (b may be a module or a
    name inside a -- whoever imports decides). Top-level statements and the ones inside functions
    alike: a lazy import is exactly the one a packaged build fails on late."""
    root = Path(package_dir) if package_dir is not None else package_directory()
    names = set()
    for path in sorted(root.rglob("*.py")):
        if path.relative_to(root).parts[0] == "tests":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names
                             if alias.name != "*")
    return sorted(n for n in names if n != PACKAGE and not n.startswith(PACKAGE + "."))


def _is_a_module_here(name: str) -> bool:
    """Is `name` a module this Python can find? False for a class or function named by a
    `from a import b`, and for a module of another platform (fcntl on Windows)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _import_each(names: Iterable[str]) -> List[str]:
    """Import every name; the ones that would not, each with its reason."""
    problems = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as e:                       # the reason IS the finding
            problems.append(f"{name} ({type(e).__name__}: {e})")
    return problems


def manifest_path() -> Path:
    return package_directory() / MANIFEST_NAME


def read_manifest() -> Dict[str, List[str]]:
    """The packaged build's manifest. Raises CheckFailed, saying so, when it is not usable."""
    path = manifest_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"modules": [str(n) for n in data["modules"]],
                "imports": [str(n) for n in data["imports"]]}
    except FileNotFoundError:
        raise CheckFailed(f"{MANIFEST_NAME} is not in the build ({path}): the spec writes it and "
                          "lists it in `datas`") from None
    except Exception as e:
        raise CheckFailed(f"{path} cannot be read ({type(e).__name__}: {e})") from None


def _loaded_modules() -> List[str]:
    """Every module this process has loaded from a FILE -- what the from-source check needed,
    which the spec then names to PyInstaller one by one (`hiddenimports`), so nothing the check
    exercised depends on static analysis having seen it. Built-in and frozen modules need no
    naming; neither does what only the build environment loads."""
    out = []
    for name, module in sorted(sys.modules.items()):
        top = name.split(".")[0]
        if (module is None or name in ("__main__", "__mp_main__") or top in _NOT_THE_APPS
                or top.startswith("__editable__") or name.startswith(PACKAGE + ".tests")):
            continue
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin and os.path.isfile(origin):
            out.append(name)
    return out


def write_manifest(path: Union[str, Path], modules: Sequence[str],
                   imports: Sequence[str]) -> Dict[str, List[str]]:
    """The manifest a packaged build is checked against, written from SOURCE once the whole
    check has passed there."""
    manifest = {"modules": list(modules), "imports": list(imports), "loaded": _loaded_modules()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


# =============================================================================================
# the steps -- each returns its detail line (or (status, detail)); raising is a FAIL

def environment() -> str:
    """The environment self-check the app has run at every launch since the work incident of
    2026-07-10 (a different pydantic at work silently changed validation behaviour and killed a
    full day of processing): the libraries import, and a NaN is still coerced to None. Returns
    the versions; raises when the environment is not the tested one."""
    import pydantic, numpy, pandas, sqlalchemy, matplotlib, customtkinter
    from laser_trim_analyzer.core.models import TrackData, AnalysisStatus
    # The coercion hook rightly WARNS when it drops a NaN (2026-08-31, the linearity-magnitude
    # fix made that silencer loud). This probe feeds it a deliberate NaN, so mute the models
    # logger for just this line -- otherwise every launch opens with an alarming warning the
    # self-check itself caused, and real drop warnings lose their signal value.
    models_logger = logging.getLogger("laser_trim_analyzer.core.models")
    models_logger.disabled = True
    try:
        track = TrackData(track_id="_env", travel_length=1.0, linearity_spec=0.01,
                          status=AnalysisStatus.PASS, linearity_error=float("nan"))
    finally:
        models_logger.disabled = False
    if track.linearity_error is not None:
        raise RuntimeError("NaN coercion inactive")
    return (f"pydantic {pydantic.VERSION}, numpy {numpy.__version__}, pandas "
            f"{pandas.__version__}, sqlalchemy {sqlalchemy.__version__}, matplotlib "
            f"{matplotlib.__version__}, customtkinter "
            f"{getattr(customtkinter, '__version__', '?')}")


def _step_build() -> Tuple[str, str]:
    info = read_build_info()
    if info is None:
        return NOTE, (f"no {BUILD_INFO_NAME} beside the app (running from source, or built "
                      "without scripts\\build_exe.ps1)")
    return OK, describe_build(info)


def _step_data() -> Tuple[str, str]:
    """Which database the app WOULD open -- looked at, never opened. Not a pass or a fail of the
    build: it is the answer to "is this copy reading the data beside it?"."""
    data_dir = app_directory() / "data"
    beside = data_dir / "analysis.db"
    path = Path(_config.get_config().database.path)
    if os.path.isfile(path):
        said = f"the app would open {path} ({os.path.getsize(path) / 1e9:.2f} GB)"
    else:
        said = f"the app would create a new, EMPTY database at {path}"
    same = (os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(beside)))
    if not same:
        said += " -- NOT the one beside the app: config.yaml names it"
    if not os.path.isdir(data_dir):
        said = f"no data folder beside the app yet ({data_dir}); " + said
    return NOTE, said


def _step_fonts() -> str:
    from laser_trim_analyzer.gui.v6 import font_loader
    missing = [n for n in font_loader.FILES if not (font_loader.FONT_DIR / n).is_file()]
    if missing:
        raise CheckFailed(f"missing from {font_loader.FONT_DIR}: {', '.join(missing)}")
    loaded = font_loader.load_bundled_fonts()
    unread = [n for n in font_loader.FILES if not loaded.get(n, {}).get("matplotlib")]
    if unread:
        raise CheckFailed(f"matplotlib could not read {', '.join(unread)}")
    said = f"{len(font_loader.FILES)} bundled font files present and read by matplotlib"
    if sys.platform.startswith("win"):
        declined = [n for n in font_loader.FILES if not loaded.get(n, {}).get("tk")]
        said += ("; loaded for the window" if not declined else
                 f"; Windows declined {len(declined)} of them for the window -- the app then "
                 "uses its fallback font")
    return said


def _step_modules() -> str:
    names = read_manifest()["modules"] if is_frozen() else app_modules()
    problems = _import_each(names)
    if problems:
        raise CheckFailed(f"{len(problems)} of {len(names)} app modules did not import: "
                          + "; ".join(problems))
    return f"{len(names)} app modules imported"


def checked_imports() -> Tuple[List[str], List[str]]:
    """(the modules the app's import statements name that import here, the ones that are here
    but do NOT import) -- from source. A name that is no module at all on this computer (a
    class, a module of another platform) is in neither."""
    good, problems = [], []
    for name in named_imports():
        if not _is_a_module_here(name):
            continue
        failed = _import_each([name])
        if failed:
            problems.extend(failed)
        else:
            good.append(name)
    return good, problems


def _step_libraries() -> str:
    if is_frozen():
        names = read_manifest()["imports"]
        problems = _import_each(names)
    else:
        names, problems = checked_imports()
        names = names + problems
    if problems:
        raise CheckFailed(f"{len(problems)} of the {len(names)} modules the app imports did "
                          "not import: " + "; ".join(problems))
    return f"{len(names)} modules named by the app's import statements, all imported"


def _step_window_toolkit() -> str:
    """Tcl/Tk and CustomTkinter's own files. A Tk root is created and withdrawn before it is
    ever drawn, then destroyed: nothing appears on screen."""
    import tkinter
    import tkinter.font
    import customtkinter
    assets = Path(customtkinter.__file__).resolve().parent / "assets"
    needed = ("themes/blue.json", "fonts/CustomTkinter_shapes_font.otf")
    missing = [n for n in needed if not (assets / n).is_file()]
    if missing:
        raise CheckFailed(f"CustomTkinter's files are missing from {assets}: "
                          + ", ".join(missing))
    root = tkinter.Tk()
    try:
        root.withdraw()
        patch = root.tk.call("info", "patchlevel")
        plex = "IBM Plex Sans" in tkinter.font.families(root)
    finally:
        root.destroy()
    return (f"Tk {patch} starts, CustomTkinter {getattr(customtkinter, '__version__', '?')} "
            f"has its themes and fonts; the window will use "
            + ("the bundled IBM Plex font" if plex else "its fallback font"))


def _step_charts() -> str:
    """A chart drawn with no window, in every format the app exports (PNG, PDF, SVG) --
    matplotlib finds those backends by NAME when a figure is saved."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import matplotlib.backends.backend_tkagg  # noqa: F401 -- the canvas every chart widget uses
    fig = Figure(figsize=(3, 2))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot([0.0, 0.5, 1.0], [0.001, -0.002, 0.0005])
    ax.set_title("self-check")
    sizes = []
    for fmt in ("png", "pdf", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt)
        if not buf.getvalue():
            raise CheckFailed(f"saving a chart as {fmt} wrote nothing")
        sizes.append(f"{fmt} {len(buf.getvalue()):,} bytes")
    return "a chart drawn and saved as " + ", ".join(sizes)


def _a_tiny_xls(cells: Dict[Tuple[int, int], float]) -> bytes:
    """A real Excel 2 worksheet (BOF, one NUMBER record per cell, EOF): the smallest .xls there
    is. pandas recognises it by its first bytes and hands it to xlrd, as it does a trim file."""
    out = struct.pack("<HHHH", 0x0009, 4, 0x0007, 0x0010)
    for (row, col), value in cells.items():
        out += struct.pack("<HHHH3sd", 0x0003, 15, row, col, b"\0\0\0", value)
    return out + struct.pack("<HH", 0x000A, 0)


def _step_excel() -> str:
    """The two Excel engines, each found by NAME at run time: xlrd reads an .xls exactly as the
    parsers ask for it (`pd.ExcelFile(BytesIO(...))`, no engine named), openpyxl writes and
    reads an .xlsx as the exports do."""
    import pandas as pd
    want = [[1.5, -0.25], [3.0, 0.125]]
    xls = _a_tiny_xls({(r, c): want[r][c] for r in range(2) for c in range(2)})
    book = pd.ExcelFile(io.BytesIO(xls))
    got = pd.read_excel(book, sheet_name=0, header=None).values.tolist()
    if book.engine != "xlrd" or got != want:
        raise CheckFailed(f"an .xls was read by {book.engine!r} as {got}, not by xlrd as {want}")
    frame = pd.DataFrame({"position": [0.0, 0.5, 1.0], "error": [0.001, -0.002, 0.0005]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="check", index=False)
    back = pd.read_excel(io.BytesIO(buf.getvalue()), sheet_name="check")
    if not back.equals(frame):
        raise CheckFailed("an .xlsx written with openpyxl did not read back as written")
    import openpyxl
    import xlrd
    return (f"an .xls read with xlrd {xlrd.__version__}, an .xlsx written and read with "
            f"openpyxl {openpyxl.__version__}")


def _step_ml() -> str:
    """The estimators the app trains and scores with, and the scipy calls the analysis makes,
    on a few invented numbers: compiled extensions import their neighbours from C, where no
    static analysis sees them -- they only show when the code runs."""
    import pickle
    import numpy as np
    from scipy import optimize, stats
    from scipy.signal import butter, filtfilt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, cross_val_predict,
                                         cross_val_score, train_test_split)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    import scipy
    import sklearn

    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4))
    y = (X[:, 0] + 0.3 * rng.normal(size=90) > 0).astype(int)
    groups = np.arange(90) % 9

    # the failure predictor (ml/predictor.py)
    scaled = StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(scaled, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    forest = RandomForestClassifier(n_estimators=10, max_depth=4, class_weight="balanced",
                                    random_state=42, n_jobs=-1).fit(x_train, y_train)
    predicted = forest.predict(x_test)
    for metric in (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix):
        metric(y_test, predicted)
    roc_auc_score(y_test, forest.predict_proba(x_test)[:, 1])
    cross_val_score(forest, scaled, y, groups=groups, cv=GroupKFold(n_splits=3),
                    scoring="accuracy")
    next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(scaled, y, groups))
    # How a trained predictor is stored. Trusted bytes: this process pickled them on this line.
    again = pickle.loads(pickle.dumps(forest))
    if not np.array_equal(again.predict(x_test), predicted):
        raise CheckFailed("a pickled forest did not predict what it predicted before")

    # the composite trim-risk model (ml/composite_risk.py)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000))
    cross_val_predict(pipe, X, y, cv=GroupKFold(3), groups=groups, method="predict_proba")

    # scipy as the analysis and the profiler call it
    b, a = butter(2, 0.2, btype="low")
    filtfilt(b, a, X[:, 0])
    best = optimize.minimize(lambda v: float((v[0] - 1.0) ** 2), [0.0])
    stats.norm.ppf(0.975)
    stats.pearsonr(X[:, 0], X[:, 1])
    stats.skew(X[:, 0])
    stats.kurtosis(X[:, 0])
    if not best.success or abs(float(best.x[0]) - 1.0) > 1e-3:
        raise CheckFailed(f"scipy.optimize.minimize found {best.x} for a minimum at 1.0")
    return (f"a forest and a logistic pipeline trained, scored and pickled (scikit-learn "
            f"{sklearn.__version__}, joblib {joblib.__version__}); the filter, the optimiser "
            f"and the statistics ran (scipy {scipy.__version__})")


def _step_database() -> str:
    """A NEW database in a temp folder, built and read back by the app's own manager -- the
    SQLite dialect (found by name from the URL), every table and every start-up migration. A
    path is always named: this never opens the app's own database."""
    from sqlalchemy import text
    from laser_trim_analyzer.database.manager import DatabaseManager
    with tempfile.TemporaryDirectory(prefix="lta_check_", ignore_cleanup_errors=True) as td:
        db = DatabaseManager(Path(td) / "self_check.db")
        try:
            with db.session() as session:
                tables = session.execute(
                    text("select count(*) from sqlite_master where type='table'")).scalar()
                version = session.execute(text("select sqlite_version()")).scalar()
        finally:
            db.close()
    if not tables:
        raise CheckFailed("the new database has no tables")
    return f"a new database built and read back in a temp folder ({tables} tables, SQLite {version})"


STEPS: Tuple[Tuple[str, Callable[[], object]], ...] = (
    ("environment", environment),
    ("build", _step_build),
    ("data", _step_data),
    ("fonts", _step_fonts),
    ("app modules", _step_modules),
    ("libraries", _step_libraries),
    ("window toolkit", _step_window_toolkit),
    ("charts", _step_charts),
    ("excel", _step_excel),
    ("ml", _step_ml),
    ("database", _step_database),
)


# =============================================================================================
# the run

class _Heard(logging.Handler):
    """What the app's own loggers said at WARNING or above while a step ran."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.said: List[str] = []

    def emit(self, record):
        try:
            self.said.append(f"{record.levelname} {record.name}: {record.getMessage()}")
        except Exception:
            pass


def _print(line: str) -> None:
    """To the console when there is one (a windowed build has none; a console in an old code
    page may not have every character)."""
    try:
        print(line, flush=True)
    except Exception:
        try:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass


_AUTO = object()


def run(out: Optional[Callable[[str], None]] = None, result_path: object = _AUTO,
        steps: Optional[Sequence[Tuple[str, Callable[[], object]]]] = None,
        manifest_out: Union[str, Path, None] = None) -> int:
    """Run every step; 0 when none failed, else 1.

    `out` takes each line (default: the console). `result_path`: where the same lines are also
    written -- by default `check_result.txt` beside the app in a packaged build, nowhere from
    source. `manifest_out` (from source only, for the build): where to write the manifest, once
    every step has passed.
    """
    lines: List[str] = []
    out = out or _print

    def say(line: str) -> None:
        lines.append(line)
        out(line)

    if result_path is _AUTO:
        result_path = app_directory() / CHECK_RESULT_NAME if is_frozen() else None
    say("Laser Trim Analyzer self-check -- "
        + ("a packaged build in " if is_frozen() else "running from source in ")
        + str(app_directory()))
    heard = _Heard()
    root_logger = logging.getLogger()
    root_logger.addHandler(heard)
    failed: List[str] = []
    try:
        for name, step in (STEPS if steps is None else steps):
            del heard.said[:]
            where: List[str] = []
            try:
                result = step()
                status, detail = result if isinstance(result, tuple) else (OK, result)
            except CheckFailed as e:
                status, detail = FAIL, str(e)
            except Exception as e:
                status, detail = FAIL, f"{type(e).__name__}: {e}"
                where = [f"{Path(f.filename).name}:{f.lineno} in {f.name}"
                         for f in traceback.extract_tb(e.__traceback__)[-3:]]
            say(f"{status:<5}{name:<15} {detail}")
            for frame in where:
                say(f"{'':<21}at {frame}")
            for said in heard.said:
                say(f"{'':<21}log: {said}")
            if status == FAIL:
                failed.append(name)
    finally:
        root_logger.removeHandler(heard)

    if manifest_out is not None:
        if is_frozen():
            say("note: --write-manifest is for a run from source (the build); ignored here")
        elif failed:
            say("no manifest written: the check has to pass from source before a build")
        else:
            imports, _problems = checked_imports()
            manifest = write_manifest(manifest_out, app_modules(), imports)
            say(f"manifest: {manifest_out} ({len(manifest['modules'])} app modules, "
                f"{len(manifest['imports'])} named imports, {len(manifest['loaded'])} modules "
                "loaded)")
    say(OK_LINE if not failed else f"{INCOMPLETE_LINE}: {', '.join(failed)}")
    if result_path:
        try:
            # utf-8 WITH a signature: PowerShell 5.1's Get-Content reads it right only then.
            Path(result_path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        except OSError as e:
            out(f"(could not write {result_path}: {e})")
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`--check [--write-manifest PATH]` -- what the entry point hands over."""
    argv = list(sys.argv[1:] if argv is None else argv)
    manifest_out = None
    if "--write-manifest" in argv:
        at = argv.index("--write-manifest")
        if at + 1 >= len(argv):
            _print("usage: --check --write-manifest PATH")
            return 2
        manifest_out = argv[at + 1]
    return run(manifest_out=manifest_out)
