"""`--check` -- the app checks itself without a window -- and the build stamp (2026-09-30).

The packaged build is made on the work laptop, never here, so what CAN be proven here is:

  * from source, `python -m laser_trim_analyzer --check` runs every step, prints one line each,
    ends `PACKAGED BUILD OK`, exits 0 -- and writes NOTHING into the app's folder: no `data`
    folder, no log (the build's output must never contain a `data` folder);
  * the same check run AS A PACKAGED BUILD WOULD SEE ITSELF -- a child Python with `sys.frozen`
    set, `sys.executable` a LaserTrimAnalyzer.exe in a temp folder, the package copied into an
    `_internal` folder beside it with the manifest the build writes, started through the real
    launcher -- passes, writes `check_result.txt` beside the .exe, and prints the build stamp;
  * it can FAIL, by name: a font file missing, an app module or a library that is in the
    manifest but not in the build, no manifest at all, a step that raises;
  * it never opens the app's own database and never turns on the app's default-database switch;
  * the build stamp reader never raises: present, absent, or malformed.

The from-source run happens in a COPY of the package inside a temp "checkout", so the app's
folder is that temp folder -- this checkout's `data` is never looked at.
"""
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from laser_trim_analyzer import config as cfg
from laser_trim_analyzer import selfcheck

REPO = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO / "src" / "laser_trim_analyzer"
LAUNCHER = REPO / "packaging" / "laser_trim_v6_launcher.py"

STEP_NAMES = ["environment", "build", "data", "fonts", "app modules", "libraries",
              "window toolkit", "charts", "excel", "ml", "database"]


def _copy_package(into: Path) -> None:
    shutil.copytree(PACKAGE_SRC, into / "laser_trim_analyzer",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _env(src: Path) -> dict:
    return dict(os.environ, PYTHONPATH=str(src), PYTHONDONTWRITEBYTECODE="1")


def _tree(root: Path, skip=()) -> list:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if not any(str(p.relative_to(root)).startswith(s) for s in skip))


def _status(line: str) -> str:
    return line.split()[0] if line.strip() else ""


# ---- from source ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def from_source(tmp_path_factory):
    """`python -m laser_trim_analyzer --check --write-manifest ...` in a temp checkout that
    holds a copy of the package: (the checkout, the manifest path, the finished process)."""
    root = (tmp_path_factory.mktemp("checkout")).resolve()
    _copy_package(root / "src")
    manifest = root / "build" / "packaged_manifest.json"
    r = subprocess.run([sys.executable, "-B", "-m", "laser_trim_analyzer", "--check",
                        "--write-manifest", str(manifest)],
                       cwd=root, env=_env(root / "src"), capture_output=True, text=True,
                       timeout=240)
    return root, manifest, r


def test_from_source_every_step_passes_and_the_last_line_says_so(from_source):
    root, _manifest, r = from_source
    lines = r.stdout.splitlines()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert lines[-1] == "PACKAGED BUILD OK", lines[-5:]
    assert lines[0] == f"Laser Trim Analyzer self-check -- running from source in {root}"
    steps = [ln for ln in lines if _status(ln) in ("ok", "note", "FAIL")]
    assert [ln[5:20].strip() for ln in steps] == STEP_NAMES, steps
    assert [_status(ln) for ln in steps] == ["ok", "note", "note"] + ["ok"] * 8, steps


def test_from_source_the_check_writes_nothing_into_the_apps_folder(from_source):
    """No `data` folder, no log file, no result file: only the manifest the build asked for."""
    root, manifest, r = from_source
    assert r.returncode == 0, r.stdout[-2000:]
    assert _tree(root, skip=("src",)) == ["build", "build/packaged_manifest.json"]
    data_line = next(ln for ln in r.stdout.splitlines() if ln.startswith("note data"))
    assert f"no data folder beside the app yet ({root / 'data'})" in data_line, data_line
    assert f"would create a new, EMPTY database at {root / 'data' / 'analysis.db'}" in data_line


def test_the_manifest_lists_the_apps_modules_its_named_imports_and_what_was_loaded(from_source):
    _root, manifest, r = from_source
    assert r.returncode == 0, r.stdout[-2000:]
    got = json.loads(manifest.read_text())
    assert got["modules"] == selfcheck.app_modules(PACKAGE_SRC)
    assert "laser_trim_analyzer.gui.v6.app" in got["modules"]
    assert "laser_trim_analyzer.selfcheck" in got["modules"]
    assert not [m for m in got["modules"] if ".tests" in m or m.endswith("__main__")]
    # named by an import statement inside a FUNCTION -- the lazy ones a build fails on late
    for lazy in ("sklearn.linear_model", "matplotlib.backends.backend_pdf", "openpyxl.styles"):
        assert lazy in got["imports"], lazy
    # `from sklearn.ensemble import RandomForestClassifier` names a class, not a module
    assert "sklearn.ensemble" in got["imports"]
    assert "sklearn.ensemble.RandomForestClassifier" not in got["imports"]
    assert not [m for m in got["imports"] if m.startswith("laser_trim_analyzer")]
    # found by NAME at run time, so no import statement of the app's parsers names them
    for by_name in ("xlrd", "matplotlib.backends.backend_svg", "matplotlib.backends.backend_agg",
                    "sqlalchemy.dialects.sqlite.pysqlite", "openpyxl"):
        assert by_name in got["loaded"], by_name
    assert set(got["modules"]) <= set(got["loaded"])
    # every named import that is a FILE was loaded; the rest are built into the interpreter
    import importlib.util
    not_files = {importlib.util.find_spec(m).origin
                 for m in set(got["imports"]) - set(got["loaded"])}
    assert not_files <= {"built-in", "frozen"}, not_files
    assert not [m for m in got["loaded"]
                if m.split(".")[0] in ("__main__", "pytest", "_pytest", "PyInstaller")]


# ---- as a packaged build sees itself -----------------------------------------------------------

_AS_PACKAGED = ("import runpy, sys\n"
                "sys.frozen = True\n"
                "sys.executable = sys.argv[1]\n"
                "launcher = sys.argv[2]\n"
                "sys.argv = [sys.argv[1]] + sys.argv[3:]\n"
                "runpy.run_path(launcher, run_name='__main__')\n")


def _packaged_app(tmp_path, manifest) -> Path:
    """A temp LaserTrimAnalyzer folder laid out as PyInstaller's: the package inside
    `_internal`, the manifest beside the self-check module."""
    app = (tmp_path / "LaserTrimAnalyzer").resolve()
    _copy_package(app / "_internal")
    if manifest is not None:
        shutil.copy2(manifest, app / "_internal" / "laser_trim_analyzer" / selfcheck.MANIFEST_NAME)
    return app


def _run_packaged(app: Path, *args: str):
    return subprocess.run([sys.executable, "-B", "-c", _AS_PACKAGED,
                           str(app / "LaserTrimAnalyzer.exe"), str(LAUNCHER), *args],
                          cwd=app.parent, env=_env(app / "_internal"), capture_output=True,
                          text=True, timeout=240)


def test_as_a_packaged_build_the_check_passes_and_leaves_its_result_beside_the_exe(
        from_source, tmp_path):
    _root, manifest, built = from_source
    assert built.returncode == 0, built.stdout[-2000:]
    app = _packaged_app(tmp_path, manifest)
    (app / "build_info.txt").write_text(selfcheck.format_build_info({
        "commit": "abc1234", "commit_date": "2026-09-26", "built": "2026-10-01 09:12",
        "python": "3.12.7", "pyinstaller": "6.16.0"}))
    r = _run_packaged(app, "--check")
    lines = r.stdout.splitlines()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert lines[0] == f"Laser Trim Analyzer self-check -- a packaged build in {app}"
    assert lines[-1] == "PACKAGED BUILD OK"
    assert not [ln for ln in lines if _status(ln) == "FAIL"], lines
    # the same lines, beside the .exe, for a windowed build that has no console
    result = app / "check_result.txt"
    assert result.read_bytes()[:3] == b"\xef\xbb\xbf"          # PowerShell 5.1 reads it right
    assert result.read_text(encoding="utf-8-sig").splitlines() == lines
    # the build stamp, and the modules of the MANIFEST (not a walk of what happens to be there)
    assert ("ok   build           abc1234 of 2026-09-26; built 2026-10-01 09:12 with Python "
            "3.12.7 and PyInstaller 6.16.0") in lines
    n = len(json.loads(manifest.read_text())["modules"])
    assert f"ok   app modules     {n} app modules imported" in lines
    # and nothing else beside the .exe: above all, no `data` folder
    assert sorted(p.name for p in app.iterdir()) == ["_internal", "build_info.txt",
                                                    "check_result.txt"]


# ---- it can fail, by name ----------------------------------------------------------------------

def _lines(**kwargs):
    said = []
    code = selfcheck.run(out=said.append, **kwargs)
    return code, said


def test_a_font_missing_from_the_build_is_named(monkeypatch, tmp_path):
    from laser_trim_analyzer.gui.v6 import font_loader
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    for name in font_loader.FILES[1:]:
        shutil.copy2(font_loader.FONT_DIR / name, fonts / name)
    monkeypatch.setattr(font_loader, "FONT_DIR", fonts)
    code, said = _lines(steps=[("fonts", selfcheck._step_fonts)], result_path=None)
    assert code == 1 and said[-1] == "PACKAGED BUILD INCOMPLETE: fonts"
    assert _status(said[1]) == "FAIL" and font_loader.FILES[0] in said[1], said


@pytest.fixture
def as_packaged(monkeypatch, tmp_path):
    """This process as a packaged build, with a manifest of the test's own making."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    path = tmp_path / "bundle" / selfcheck.MANIFEST_NAME
    path.parent.mkdir()
    monkeypatch.setattr(selfcheck, "manifest_path", lambda: path)

    def write(modules=(), imports=()):
        path.write_text(json.dumps({"modules": list(modules), "imports": list(imports),
                                    "loaded": []}))
        return path
    return write


def test_an_app_module_left_out_of_the_build_is_named(as_packaged):
    as_packaged(modules=["laser_trim_analyzer.config", "laser_trim_analyzer.left_out_of_the_build"])
    code, said = _lines(steps=[("app modules", selfcheck._step_modules)], result_path=None)
    assert code == 1 and said[-1] == "PACKAGED BUILD INCOMPLETE: app modules"
    assert "1 of 2 app modules did not import" in said[1]
    assert ("laser_trim_analyzer.left_out_of_the_build (ModuleNotFoundError: No module named "
            "'laser_trim_analyzer.left_out_of_the_build')") in said[1], said[1]


def test_a_library_the_app_imports_that_is_not_in_the_build_is_named(as_packaged):
    as_packaged(imports=["json", "an_invented_library_not_in_the_build"])
    code, said = _lines(steps=[("libraries", selfcheck._step_libraries)], result_path=None)
    assert code == 1 and said[-1] == "PACKAGED BUILD INCOMPLETE: libraries"
    assert "an_invented_library_not_in_the_build (ModuleNotFoundError" in said[1], said[1]
    assert "1 of the 2 modules" in said[1]


def test_a_build_without_its_manifest_says_so_instead_of_listing_what_is_there(as_packaged):
    """The list of what MUST be there cannot come from what IS there."""
    code, said = _lines(steps=[("app modules", selfcheck._step_modules),
                               ("libraries", selfcheck._step_libraries)], result_path=None)
    assert code == 1 and said[-1] == "PACKAGED BUILD INCOMPLETE: app modules, libraries"
    assert all("packaged_manifest.json is not in the build" in ln for ln in said[1:3]), said


def test_a_step_that_raises_is_one_fail_and_the_rest_still_run(tmp_path):
    def crashes():
        raise ValueError("an invented crash")

    def warns():
        logging.getLogger("laser_trim_analyzer.invented").warning("an invented warning")
        return "said something"

    result = tmp_path / "result.txt"
    code, said = _lines(steps=[("first", lambda: "fine"), ("second", crashes),
                               ("third", lambda: (selfcheck.NOTE, "only a note")),
                               ("fourth", warns)], result_path=result)
    assert code == 1 and said[-1] == "PACKAGED BUILD INCOMPLETE: second"
    assert [_status(ln) for ln in said[1:-1] if _status(ln) in ("ok", "note", "FAIL")] == \
        ["ok", "FAIL", "note", "ok"]
    assert "ValueError: an invented crash" in said[2]
    assert any("at test_packaged_check.py:" in ln and "in crashes" in ln for ln in said), said
    assert any("log: WARNING laser_trim_analyzer.invented: an invented warning" in ln
               for ln in said), said
    assert result.read_text(encoding="utf-8-sig").splitlines() == said


def test_from_source_no_result_file_is_written_and_packaged_it_lands_beside_the_exe(
        monkeypatch, tmp_path):
    app = tmp_path / "LaserTrimAnalyzer"
    app.mkdir()
    monkeypatch.setattr(cfg, "get_app_directory", lambda: app)
    steps = [("only", lambda: "fine")]
    assert selfcheck.run(out=lambda line: None, steps=steps) == 0
    assert list(app.iterdir()) == []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert selfcheck.run(out=lambda line: None, steps=steps) == 0
    assert [p.name for p in app.iterdir()] == ["check_result.txt"]
    assert (app / "check_result.txt").read_text(encoding="utf-8-sig").splitlines()[-1] == \
        "PACKAGED BUILD OK"


def test_a_manifest_is_written_only_from_source_and_only_when_the_check_passed(
        monkeypatch, tmp_path):
    def fails():
        raise selfcheck.CheckFailed("invented")

    out = tmp_path / "m.json"
    code, said = _lines(steps=[("only", fails)], result_path=None, manifest_out=out)
    assert code == 1 and not out.exists()
    assert "no manifest written" in said[-2]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    code, said = _lines(steps=[("only", lambda: "fine")], result_path=None, manifest_out=out)
    assert code == 0 and not out.exists() and "ignored" in said[-2]


# ---- which database the app would open (looked at, never opened) -------------------------------

def test_the_data_step_says_which_database_a_handed_over_copy_would_open(monkeypatch, tmp_path):
    """Her PC: config.yaml names the owner's absolute path, the copy is beside the app. The
    check says the app would open the copy -- and repeats the config's own warning."""
    import yaml
    app = tmp_path / "her_pc" / "LaserTrimAnalyzer"
    (app / "data").mkdir(parents=True)
    (app / "data" / "analysis.db").write_bytes(b"x" * 2048)
    theirs = tmp_path / "his_pc" / "data" / "analysis.db"
    (app / "data" / "config.yaml").write_text(yaml.safe_dump({"database": {"path": str(theirs)}}))
    monkeypatch.setattr(cfg, "get_app_directory", lambda: app)
    monkeypatch.setattr(cfg, "_config", None)
    before = _tree(tmp_path)
    code, said = _lines(steps=[("data", selfcheck._step_data)], result_path=None)
    assert code == 0 and _status(said[1]) == "note"
    assert f"the app would open {app / 'data' / 'analysis.db'} (0.00 GB)" in said[1]
    assert "NOT the one beside the app" not in said[1]
    assert any("log: WARNING" in ln and "not on this computer" in ln for ln in said), said
    assert _tree(tmp_path) == before, "the check created or changed something"


def test_the_data_step_says_so_when_the_database_is_not_the_one_beside_the_app(monkeypatch,
                                                                              tmp_path):
    """The owner's own laptop: the configured path exists, so it is used -- even with a copy
    beside a packaged app. The check says which, so it cannot be mistaken for the copy."""
    import yaml
    app = tmp_path / "LaserTrimAnalyzer"
    (app / "data").mkdir(parents=True)
    (app / "data" / "analysis.db").write_bytes(b"copy")
    own = tmp_path / "dev" / "data" / "analysis.db"
    own.parent.mkdir(parents=True)
    own.write_bytes(b"the real one")
    (app / "data" / "config.yaml").write_text(yaml.safe_dump({"database": {"path": str(own)}}))
    monkeypatch.setattr(cfg, "get_app_directory", lambda: app)
    monkeypatch.setattr(cfg, "_config", None)
    code, said = _lines(steps=[("data", selfcheck._step_data)], result_path=None)
    assert code == 0
    assert f"the app would open {own}" in said[1]
    assert "NOT the one beside the app: config.yaml names it" in said[1], said[1]


# ---- the entry point ---------------------------------------------------------------------------

def _no_window(monkeypatch):
    """Neither UI can be built: a `--check` that fell through to the app would fail here, at
    once, instead of opening a real window and waiting in its event loop."""
    import laser_trim_analyzer.app as v5_mod
    import laser_trim_analyzer.gui.v6.app as v6_mod
    monkeypatch.setattr(v6_mod, "V6App", MagicMock(side_effect=AssertionError("a window")))
    monkeypatch.setattr(v5_mod, "LaserTrimApp", MagicMock(side_effect=AssertionError("a window")))


def test_main_with_check_runs_the_check_and_never_the_app_or_its_database(
        monkeypatch, tmp_path, capsys):
    from laser_trim_analyzer.__main__ import main
    from laser_trim_analyzer.database import manager as mgr
    app_dir = Path(cfg.get_app_directory())
    _no_window(monkeypatch)
    allowed = []
    monkeypatch.setattr(mgr, "allow_default_database", lambda: allowed.append(1))
    opened = []
    inner = mgr.DatabaseManager.__init__

    def recorded(self, database_path=None, *a, **k):
        opened.append(database_path)
        return inner(self, database_path, *a, **k)

    monkeypatch.setattr(mgr.DatabaseManager, "__init__", recorded)
    monkeypatch.setattr(selfcheck, "STEPS", (("environment", selfcheck.environment),
                                             ("database", selfcheck._step_database)))
    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer", "--v6", "--check"])
    before = _tree(app_dir)
    with pytest.raises(SystemExit) as stop:
        main()
    out = capsys.readouterr().out.splitlines()
    assert stop.value.code == 0 and out[-1] == "PACKAGED BUILD OK", out
    assert allowed == [] and mgr._default_database_allowed is False
    assert len(opened) == 1 and opened[0] is not None
    assert app_dir not in Path(opened[0]).resolve().parents, opened
    assert not Path(opened[0]).exists(), "the temp database was left behind"
    assert _tree(app_dir) == before, "the check wrote into the app's folder"


def test_main_with_check_exits_1_when_a_step_fails(monkeypatch, capsys):
    from laser_trim_analyzer.__main__ import main

    def fails():
        raise selfcheck.CheckFailed("invented: not in the build")

    _no_window(monkeypatch)
    monkeypatch.setattr(selfcheck, "STEPS", (("invented", fails),))
    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer", "--check"])
    with pytest.raises(SystemExit) as stop:
        main()
    assert stop.value.code == 1
    assert capsys.readouterr().out.splitlines()[-1] == "PACKAGED BUILD INCOMPLETE: invented"


def test_in_check_mode_logging_makes_no_data_folder_and_no_log_file(monkeypatch, tmp_path):
    """The build runs `LaserTrimAnalyzer.exe --check` inside the folder it is about to hand
    over: the entry point's logging must not create `data/laser_trim.log` there."""
    import laser_trim_analyzer.__main__ as entry
    app = tmp_path / "LaserTrimAnalyzer"
    app.mkdir()
    monkeypatch.setattr(entry, "get_app_directory", lambda: app)
    handlers = entry._log_handlers(check_only=True)
    assert [type(h).__name__ for h in handlers] == ["NullHandler"]
    assert list(app.iterdir()) == []


# ---- the build stamp ---------------------------------------------------------------------------

STAMP = {"commit": "abc1234", "commit_date": "2026-09-26", "built": "2026-10-01 09:12",
         "python": "3.12.7", "pyinstaller": "6.16.0"}


def test_no_stamp_beside_the_app_means_running_from_source(tmp_path):
    assert selfcheck.read_build_info(tmp_path) is None
    assert selfcheck.build_line(None) is None
    (tmp_path / "build_info.txt").mkdir()                 # a folder of that name is no stamp
    assert selfcheck.read_build_info(tmp_path) is None


@pytest.mark.parametrize("encoding, newline", [("utf-8", "\n"), ("utf-8-sig", "\r\n"),
                                               ("utf-16", "\r\n"), ("ascii", "\n")])
def test_a_stamp_is_read_however_powershell_wrote_it(tmp_path, encoding, newline):
    text = selfcheck.format_build_info(STAMP).replace("\n", newline)
    (tmp_path / "build_info.txt").write_bytes(text.encode(encoding))
    info = selfcheck.read_build_info(tmp_path)
    assert info == STAMP
    assert selfcheck.build_line(info) == "build abc1234 of 2026-09-26"
    assert selfcheck.describe_build(info) == ("abc1234 of 2026-09-26; built 2026-10-01 09:12 "
                                              "with Python 3.12.7 and PyInstaller 6.16.0")


def test_a_stamp_the_build_could_not_fill_in_says_unknown():
    assert selfcheck.format_build_info({"built": "2026-10-01 09:12"}) == (
        "commit: unknown\ncommit_date: unknown\nbuilt: 2026-10-01 09:12\n"
        "python: unknown\npyinstaller: unknown\n")


@pytest.mark.parametrize("content", [
    b"", b"\x00\x01\x02\xff\xfe\xfd" * 50, b"no colon on this line\nnor this one\n",
    b"\xff\xfe\x00", b": a value without a key\n", b"commit:\n", "commit: é\x07\n".encode(),
    b"a key with spaces: 1\n" + b"x" * 100000, b"COMMIT : abc1234 \r\n\r\n"])
def test_a_malformed_stamp_never_raises(tmp_path, content):
    (tmp_path / "build_info.txt").write_bytes(content)
    info = selfcheck.read_build_info(tmp_path)
    assert isinstance(info, dict)
    line = selfcheck.build_line(info)
    assert line.startswith("build ") and " of " in line
    assert all(ch.isprintable() for ch in line), repr(line)
    assert selfcheck.describe_build(info)
    if content.startswith(b"COMMIT"):
        assert line == "build abc1234 of unknown"


def test_a_stamp_that_cannot_be_read_is_an_empty_one_not_an_exception(tmp_path, monkeypatch):
    (tmp_path / "build_info.txt").write_text("commit: abc1234\n")
    import builtins
    real_open = builtins.open

    def refused(path, *a, **k):
        if str(path).endswith("build_info.txt"):
            raise PermissionError(13, "invented: access denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", refused)
    assert selfcheck.read_build_info(tmp_path) == {}
    assert selfcheck.build_line({}) == "build unknown of unknown"


def test_the_stamp_is_looked_for_in_the_apps_own_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "get_app_directory", lambda: tmp_path)
    assert selfcheck.read_build_info() is None
    (tmp_path / "build_info.txt").write_text(selfcheck.format_build_info(STAMP))
    assert selfcheck.read_build_info() == STAMP


def _start_the_app(monkeypatch, tmp_path):
    import laser_trim_analyzer.config as cmod
    import laser_trim_analyzer.gui.v6.app as v6_mod
    from laser_trim_analyzer.__main__ import main
    from laser_trim_analyzer.config import Config
    monkeypatch.setattr(v6_mod, "V6App", MagicMock())
    config = Config()
    config.database.path = tmp_path / "t.db"
    monkeypatch.setattr(cmod, "get_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["laser_trim_analyzer", "--v6"])
    main()


def test_the_app_logs_its_build_once_at_start_when_it_has_a_stamp(monkeypatch, tmp_path, caplog):
    app = tmp_path / "LaserTrimAnalyzer"
    app.mkdir()
    monkeypatch.setattr(cfg, "get_app_directory", lambda: app)
    (app / "build_info.txt").write_text(selfcheck.format_build_info(STAMP))
    with caplog.at_level(logging.INFO):
        _start_the_app(monkeypatch, tmp_path)
    said = [r.getMessage() for r in caplog.records if r.name == "laser_trim_analyzer.__main__"]
    assert said.count("build abc1234 of 2026-09-26") == 1, said


def test_from_source_the_app_logs_nothing_about_a_build(monkeypatch, tmp_path, caplog):
    app = tmp_path / "checkout"
    app.mkdir()
    monkeypatch.setattr(cfg, "get_app_directory", lambda: app)
    with caplog.at_level(logging.INFO):
        _start_the_app(monkeypatch, tmp_path)
    said = [r.getMessage() for r in caplog.records if r.name == "laser_trim_analyzer.__main__"]
    assert said and not [m for m in said if m.startswith("build ")], said
