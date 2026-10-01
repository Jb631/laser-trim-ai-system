"""The files the packaged build is made from: the launcher, the PyInstaller spec, the build
script and its Python helper (2026-09-30).

None of them can be RUN for real here -- PyInstaller is not installed on this machine and must
not be, and there is no PowerShell -- so each is held to what can be checked without a build:

  * the launcher runs as it is: freeze_support first, `--v6` forced, other arguments through;
  * the spec is executed with stand-ins for PyInstaller's names (in a temp copy of packaging/
    and the package, so its own from-source check never looks at this checkout's `data`): every
    path it names exists, every hidden import it names imports, it is a ONEDIR windowed build
    named LaserTrimAnalyzer, and it makes no build when the from-source check fails;
  * the PowerShell script is read as text: plain ASCII, balanced, every variable assigned, every
    file and helper command it names real, no data copied or deleted;
  * the helper is plain Python and is simply tested: the stamp it writes is the stamp the app
    reads, the READ ME says what it must, and "does PyInstaller support this Python" is right.
"""
import ast
import importlib
import importlib.util
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from laser_trim_analyzer import selfcheck

REPO = Path(__file__).resolve().parents[1]
PACKAGING = REPO / "packaging"
LAUNCHER = PACKAGING / "laser_trim_v6_launcher.py"
SPEC = PACKAGING / "laser_trim_v6.spec"
SCRIPT = REPO / "scripts" / "build_exe.ps1"
PACKAGE_SRC = REPO / "src" / "laser_trim_analyzer"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- the launcher -----------------------------------------------------------------------------

def _launch(monkeypatch, argv):
    import laser_trim_analyzer.__main__ as entry
    order, seen = [], {}
    monkeypatch.setattr(multiprocessing, "freeze_support", lambda: order.append("freeze_support"))

    def main():
        order.append("main")
        seen["argv"] = list(sys.argv)
        seen["stdout"], seen["stderr"] = sys.stdout, sys.stderr

    monkeypatch.setattr(entry, "main", main)
    monkeypatch.setattr(sys, "argv", list(argv))
    _load(LAUNCHER, "launcher_under_test").run()
    return order, seen


def test_the_launcher_asks_freeze_support_first_and_forces_the_v6_ui(monkeypatch):
    order, seen = _launch(monkeypatch, ["LaserTrimAnalyzer.exe"])
    assert order == ["freeze_support", "main"]
    assert seen["argv"] == ["LaserTrimAnalyzer.exe", "--v6"]


def test_the_launcher_passes_other_arguments_through_and_adds_v6_only_once(monkeypatch):
    _order, seen = _launch(monkeypatch, ["LaserTrimAnalyzer.exe", "--check"])
    assert seen["argv"] == ["LaserTrimAnalyzer.exe", "--v6", "--check"]
    _order, seen = _launch(monkeypatch, ["LaserTrimAnalyzer.exe", "--check", "--v6"])
    assert seen["argv"] == ["LaserTrimAnalyzer.exe", "--check", "--v6"]


def test_the_launcher_gives_a_windowed_build_somewhere_to_write(monkeypatch):
    """PyInstaller's windowed bootloader leaves sys.stdout and sys.stderr as None."""
    with monkeypatch.context() as windowed:
        windowed.setattr(sys, "stdout", None)
        windowed.setattr(sys, "stderr", None)
        _order, seen = _launch(windowed, ["LaserTrimAnalyzer.exe"])
        opened = [seen["stdout"], seen["stderr"]]
    try:
        for stream in opened:
            assert stream is not None
            stream.write("nothing raises")
    finally:
        for stream in opened:
            stream.close()


def test_the_launcher_imports_the_app_only_after_freeze_support():
    """Static: a spawned child must reach freeze_support() before the app's entry module is
    imported -- that import sets logging up and creates data/ beside the .exe."""
    tree = ast.parse(LAUNCHER.read_text())
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not [n for n in top_level if "laser_trim_analyzer" in ast.unparse(n)]
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert ast.unparse(run.body[0]) == "multiprocessing.freeze_support()"
    assert "if __name__ == '__main__':\n    run()" in ast.unparse(tree)


# ---- the spec ---------------------------------------------------------------------------------

class _Recorded:
    """Stands in for one of PyInstaller's build classes: remembers how it was called."""

    def __init__(self, calls, kind):
        self.calls, self.kind = calls, kind

    def __call__(self, *args, **kwargs):
        made = types.SimpleNamespace(kind=self.kind, args=args, kwargs=kwargs,
                                     pure=("pure",), scripts=("scripts",),
                                     binaries=("binaries",), datas=("datas",))
        self.calls.append(made)
        return made


def _collect_data_files(package):
    """What PyInstaller's helper returns: (file, folder inside the build) for every file of the
    installed package that is not Python."""
    root = Path(importlib.util.find_spec(package).origin).parent
    return [(str(p), str(Path(package) / p.relative_to(root).parent))
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts]


def _run_spec(checkout: Path, work: Path, console=None, check=None):
    """Execute the spec in `checkout` with stand-ins for PyInstaller: the recorded calls."""
    fake = types.ModuleType("PyInstaller")
    fake.utils = types.ModuleType("PyInstaller.utils")
    fake.utils.hooks = types.ModuleType("PyInstaller.utils.hooks")
    fake.utils.hooks.collect_data_files = _collect_data_files
    names = {"PyInstaller": fake, "PyInstaller.utils": fake.utils,
             "PyInstaller.utils.hooks": fake.utils.hooks}
    saved_modules = {n: sys.modules.get(n) for n in names}
    saved_env, saved_run = os.environ.get("LTA_CONSOLE"), subprocess.run
    calls = []
    namespace = {"SPECPATH": str(checkout / "packaging"), "workpath": str(work),
                 "Analysis": _Recorded(calls, "Analysis"), "PYZ": _Recorded(calls, "PYZ"),
                 "EXE": _Recorded(calls, "EXE"), "COLLECT": _Recorded(calls, "COLLECT")}
    spec = checkout / "packaging" / "laser_trim_v6.spec"
    try:
        sys.modules.update(names)
        if console is None:
            os.environ.pop("LTA_CONSOLE", None)
        else:
            os.environ["LTA_CONSOLE"] = console
        if check is not None:
            subprocess.run = check
        exec(compile(spec.read_text(), str(spec), "exec"), namespace)
    finally:
        subprocess.run = saved_run
        if saved_env is None:
            os.environ.pop("LTA_CONSOLE", None)
        else:
            os.environ["LTA_CONSOLE"] = saved_env
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return {c.kind: c for c in calls}, calls, namespace


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The spec, executed once for real (its from-source check included) in a temp checkout."""
    checkout = tmp_path_factory.mktemp("spec_checkout").resolve()
    shutil.copytree(PACKAGING, checkout / "packaging",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PACKAGE_SRC, checkout / "src" / "laser_trim_analyzer",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    by_kind, calls, namespace = _run_spec(checkout, checkout / "build" / "laser_trim_v6")
    return checkout, by_kind, calls, namespace


def test_the_spec_is_a_onedir_windowed_build_named_laser_trim_analyzer(built):
    _checkout, by_kind, calls, _ns = built
    assert [c.kind for c in calls] == ["Analysis", "PYZ", "EXE", "COLLECT"]
    exe, coll = by_kind["EXE"], by_kind["COLLECT"]
    assert exe.kwargs["name"] == "LaserTrimAnalyzer" and coll.kwargs["name"] == "LaserTrimAnalyzer"
    assert exe.kwargs["console"] is False and exe.kwargs["upx"] is False
    # ONEDIR: the binaries and data go to COLLECT, never into the .exe (that would be onefile)
    assert exe.kwargs["exclude_binaries"] is True
    assert ("binaries",) not in exe.args and ("datas",) not in exe.args
    assert ("binaries",) in coll.args and ("datas",) in coll.args and coll.kwargs["upx"] is False


def test_every_path_the_spec_names_exists(built):
    checkout, by_kind, _calls, _ns = built
    analysis = by_kind["Analysis"]
    (script,) = analysis.args[0]
    assert script == str(checkout / "packaging" / "laser_trim_v6_launcher.py")
    assert analysis.kwargs["pathex"] == [str(checkout / "src")]
    for path in [script, *analysis.kwargs["pathex"]]:
        assert os.path.isabs(path) and os.path.exists(path), path
    datas = analysis.kwargs["datas"]
    assert datas
    for source, folder in datas:
        assert os.path.isabs(source) and os.path.isfile(source), source
        assert not os.path.isabs(folder) and ".." not in Path(folder).parts, folder


def test_the_spec_bundles_everything_the_app_reads_from_beside_its_code(built):
    """The fonts land where font_loader looks for them, the manifest where the self-check does,
    CustomTkinter's files where CustomTkinter does -- and no non-Python file of the package is
    left out."""
    from laser_trim_analyzer.gui.v6 import font_loader
    checkout, by_kind, _calls, _ns = built
    landed = {(Path(folder) / Path(source).name).as_posix()
              for source, folder in by_kind["Analysis"].kwargs["datas"]}
    fonts_at = Path(font_loader.FONT_DIR).relative_to(PACKAGE_SRC.parent).as_posix()
    for name in font_loader.FILES:
        assert f"{fonts_at}/{name}" in landed, name
    assert f"laser_trim_analyzer/{selfcheck.MANIFEST_NAME}" in landed
    assert "customtkinter/assets/themes/blue.json" in landed
    assert "customtkinter/assets/fonts/CustomTkinter_shapes_font.otf" in landed
    package = checkout / "src" / "laser_trim_analyzer"
    not_python = {p.relative_to(package.parent).as_posix() for p in package.rglob("*")
                  if p.is_file() and p.suffix not in (".py", ".pyc") and p.name != ".DS_Store"}
    assert not_python <= landed, sorted(not_python - landed)


def test_every_hidden_import_the_spec_names_can_be_imported(built):
    _checkout, by_kind, _calls, namespace = built
    hidden = by_kind["Analysis"].kwargs["hiddenimports"]
    by_name = namespace["HIDDEN_BY_NAME"]
    for must in ("xlrd", "openpyxl", "matplotlib.backends.backend_tkagg",
                 "matplotlib.backends.backend_pdf", "matplotlib.backends.backend_svg",
                 "sqlalchemy.dialects.sqlite", "yaml"):
        assert must in by_name, must
    assert set(by_name) <= set(hidden)
    assert set(selfcheck.app_modules(PACKAGE_SRC)) <= set(hidden)
    assert len(hidden) == len(set(hidden)) > 500
    failed = []
    for name in hidden:
        try:
            importlib.import_module(name)
        except Exception as e:
            failed.append(f"{name}: {type(e).__name__}: {e}")
    assert not failed, failed[:10]


def test_the_spec_keeps_the_tests_out(built):
    _checkout, by_kind, _calls, _ns = built
    analysis = by_kind["Analysis"]
    assert {"laser_trim_analyzer.tests", "pytest"} <= set(analysis.kwargs["excludes"])
    assert not [m for m in analysis.kwargs["hiddenimports"]
                if m.startswith(("laser_trim_analyzer.tests", "pytest", "_pytest"))]


def _passed_before(built):
    """A stand-in for the spec's from-source check: hands back the manifest the real one wrote."""
    checkout = built[0]
    manifest = checkout / "build" / "laser_trim_v6" / "packaged_manifest.json"

    def run(cmd, **kwargs):
        target = Path(cmd[cmd.index("--write-manifest") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, target)
        return types.SimpleNamespace(returncode=0)
    return run


def test_lta_console_1_makes_the_console_build(built, tmp_path):
    by_kind, _calls, _ns = _run_spec(built[0], tmp_path / "work", console="1",
                                     check=_passed_before(built))
    assert by_kind["EXE"].kwargs["console"] is True
    by_kind, _calls, _ns = _run_spec(built[0], tmp_path / "work0", console="0",
                                     check=_passed_before(built))
    assert by_kind["EXE"].kwargs["console"] is False


def test_the_spec_makes_no_build_when_the_check_fails_from_source(built, tmp_path):
    asked = []

    def failed(cmd, **kwargs):
        asked.append(cmd)
        return types.SimpleNamespace(returncode=1)

    with pytest.raises(SystemExit) as stopped:
        _run_spec(built[0], tmp_path / "work", check=failed)
    assert "did not pass FROM SOURCE" in str(stopped.value)
    (cmd,) = asked
    assert cmd[:5] == [sys.executable, "-B", "-m", "laser_trim_analyzer", "--check"]


# ---- the build script (read, not run) ---------------------------------------------------------

def _code_lines(text):
    """The script's lines with comments and string contents removed (quotes kept)."""
    out = []
    for line in text.splitlines():
        kept, quote = [], None
        for ch in line:
            if quote:
                if ch == quote:
                    quote = None
                    kept.append(ch)
                continue
            if ch in "'\"":
                quote = ch
                kept.append(ch)
            elif ch == "#":
                break
            else:
                kept.append(ch)
        assert quote is None, f"a string is left open on: {line}"
        out.append("".join(kept))
    return out


def test_the_build_script_is_plain_ascii_and_balanced():
    """Windows PowerShell 5.1 reads a script without a byte-order mark in the ANSI code page: a
    typographic dash or quote in it becomes other characters, some of which PowerShell takes
    for string delimiters."""
    raw = SCRIPT.read_bytes()
    assert raw.isascii() and b"\t" not in raw
    code = "\n".join(_code_lines(raw.decode("ascii")))
    for opening, closing in ("{}", "()", "[]"):
        assert code.count(opening) == code.count(closing), opening
    depth = 0
    for ch in code:
        depth += {"{": 1, "}": -1}.get(ch, 0)
        assert depth >= 0
    assert depth == 0


def test_every_variable_the_build_script_reads_is_one_it_assigned():
    """A mistyped variable is not an error in PowerShell: it is silently empty."""
    text = SCRIPT.read_text()
    lines = [ln.split("#")[0] if ln.lstrip().startswith("#") else ln for ln in text.splitlines()]
    body = "\n".join(lines)
    used = set(re.findall(r"\$(?!env:)([A-Za-z_]\w*)", body))
    assigned = set(re.findall(r"\$([A-Za-z_]\w*)\s*=(?!=)", body))
    assigned |= set(re.findall(r"foreach\s*\(\s*\$([A-Za-z_]\w*)\s+in", body))
    assigned |= set(re.findall(r"\[switch\]\$([A-Za-z_]\w*)", body))
    automatic = {"PSScriptRoot", "LASTEXITCODE", "null", "true", "false"}
    assert used - assigned - automatic == set(), used - assigned - automatic
    assert set(re.findall(r"\$env:(\w+)", body)) == {"OS", "LTA_CONSOLE"}


def test_the_build_script_names_real_files_and_real_helper_commands():
    text = SCRIPT.read_text()
    for named in re.findall(r"(packaging\\[\w.]+)", text):
        assert (REPO / named.replace("\\", "/")).is_file(), named
    assert "packaging\\laser_trim_v6.spec" in text and ".venv\\Scripts\\python.exe" in text
    helper = (PACKAGING / "build_support.py").read_text()
    commands = set(re.findall(r"build_support\.py ([a-z-]+)", text))
    assert commands == {"check-pyinstaller", "stamp"}
    for command in commands:
        assert f'argv[:1] == ["{command}"]' in helper, command


def test_the_build_script_does_what_was_asked_and_copies_no_data():
    text = SCRIPT.read_text()
    code = "\n".join(_code_lines(text))
    # refuses outside Windows, before anything else is done
    assert code.index("$env:OS -ne ''") < code.index("Set-Location")
    # says plainly that installing PyInstaller downloads
    assert "DOWNLOADS FROM PyPI" in text and "-m pip install pyinstaller" in text
    # builds from the spec, a folder, never asking; a console build on request
    assert "-m PyInstaller packaging\\laser_trim_v6.spec --noconfirm" in text
    assert "--onefile" not in text and "[switch]$Console" in text
    assert "$env:LTA_CONSOLE = '1'" in text
    # the built app checks itself, and only the app's own verdict passes the build
    assert "-ArgumentList '--check'" in text and selfcheck.CHECK_RESULT_NAME in text
    assert f"-ne '{selfcheck.OK_LINE}'" in text
    # it never deletes, never copies data, and refuses a dist folder that holds a data folder
    for never in ("Remove-Item", "Copy-Item", "Move-Item", "-Recurse", "snapshot_db", "WithData",
                  "robocopy", "xcopy", "analysis.db-wal"):
        assert never not in code, never
    assert text.count("Test-Path -LiteralPath $distData") == 2


# ---- the helper -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def support():
    return _load(PACKAGING / "build_support.py", "build_support_under_test")


def test_the_stamp_the_build_writes_is_the_stamp_the_app_reads(support, tmp_path):
    import datetime
    written = support.stamp(tmp_path, now=datetime.datetime(2026, 10, 1, 9, 12))
    raw = (tmp_path / "build_info.txt").read_bytes()
    assert raw.isascii() and raw.count(b"\r\n") == 5 and b"\n" not in raw.replace(b"\r\n", b"")
    info = selfcheck.read_build_info(tmp_path)
    assert set(info) == set(selfcheck.BUILD_INFO_KEYS)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()
    assert head and info["commit"] in (head, head + "+changes"), info
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", info["commit_date"]), info
    assert info["built"] == "2026-10-01 09:12"
    assert info["python"] == "%d.%d.%d" % sys.version_info[:3]
    assert info["pyinstaller"] == "unknown"                    # not installed here
    assert selfcheck.build_line(info) == f"build {info['commit']} of {info['commit_date']}"
    assert written["commit"] == info["commit"]


def test_without_git_the_stamp_says_unknown_and_nothing_fails(support, tmp_path, monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("invented: git is not installed")

    monkeypatch.setattr(support.subprocess, "run", no_git)
    support.stamp(tmp_path)
    info = selfcheck.read_build_info(tmp_path)
    assert info["commit"] == "unknown" and info["commit_date"] == "unknown"
    assert selfcheck.build_line(info) == "build unknown of unknown"
    assert (tmp_path / support.README_NAME).is_file()


def test_the_read_me_says_what_she_needs_and_how_to_update(support, tmp_path):
    support.stamp(tmp_path)
    raw = (tmp_path / "READ ME FIRST.txt").read_bytes()
    assert raw.isascii() and b"\n" not in raw.replace(b"\r\n", b"")      # Notepad, any code page
    text = raw.decode("ascii").replace("\r\n", "\n")
    for said in ('The app only ever opens the "data" folder that sits beside LaserTrimAnalyzer.exe',
                 "Unzip this folder anywhere on your own computer",
                 "Do not put it in OneDrive",
                 "Double-click LaserTrimAnalyzer.exe. The first start takes a moment.",
                 "trained predictors are not\n  loaded on this computer. Nothing you view needs them.",
                 "UPDATING TO A NEWER VERSION",
                 "1. Close the app.",
                 '2. In this folder, delete everything EXCEPT the "data" folder.',
                 'so that LaserTrimAnalyzer.exe sits beside "data" again',
                 "the app upgrades the database by itself"):
        assert said in text, said
    assert text.index("STARTING IT") < text.index("UPDATING TO A NEWER VERSION")
    stamp = selfcheck.describe_build(selfcheck.read_build_info(tmp_path))
    assert f"This build: {stamp}" in text
    assert max(len(line) for line in text.splitlines()
               if not line.startswith("This build:")) <= 90


@pytest.mark.parametrize("requires, version, ok", [
    (">=3.8,<3.15", (3, 14, 2), True), ("<3.14,>=3.8", (3, 14, 2), False),
    ("<3.14,>=3.8", (3, 12, 7), True), (">=3.8, <3.14", (3, 7, 9), False),
    ("", (3, 14, 2), True), (">=3.8,!=3.9.*", (3, 9, 1), False), (">=3.8,!=3.9.*", (3, 10, 0), True),
    ("==3.12.*", (3, 12, 7), True), ("==3.12.*", (3, 13, 0), False), ("~=3.8", (3, 12, 0), True),
    ("<=3.13", (3, 13, 0), True), (">3.13", (3, 13, 0), False), ("something else", (3, 1, 0), True)])
def test_whether_a_python_is_inside_a_requires_python_line(support, requires, version, ok):
    assert support.python_satisfies(requires, version) is ok


def test_pyinstaller_is_reported_missing_unsupported_or_fine(support, monkeypatch):
    import importlib.metadata as metadata
    said = []
    assert support.check_pyinstaller(said.append) == 3            # this machine: not installed
    assert "PyInstaller is not installed" in said[-1]

    def with_requires(text):
        monkeypatch.setattr(metadata, "metadata", lambda name: {
            "Version": "9.9", "Requires-Python": text})

    with_requires("<3.5,>=3.0")
    assert support.check_pyinstaller(said.append) == 4
    this = "%d.%d.%d" % sys.version_info[:3]
    assert "PyInstaller 9.9 does not support this Python" in said[-2] and this in said[-2]
    assert "<3.5,>=3.0" in said[-2] and "downloads from PyPI" in said[-1]
    with_requires(">=3.8")
    assert support.check_pyinstaller(said.append) == 0
    assert "PyInstaller 9.9 supports this Python" in said[-1]


def test_the_helper_refuses_what_it_does_not_know(support, tmp_path, capsys):
    assert support.main(["bogus"]) == 2
    assert support.main(["stamp", str(tmp_path / "not there")]) == 2
    assert support.main(["stamp", str(tmp_path)]) == 0
    assert "wrote build_info.txt and READ ME FIRST.txt" in capsys.readouterr().out
