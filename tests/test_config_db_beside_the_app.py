"""The database beside the app wins over a path from another computer (2026-09-30).

The app is handed to a coworker as a folder plus a copy of the owner's `data` folder, config.yaml
included -- and that file can still hold `database.path` as the ABSOLUTE path the database has on
the computer that wrote it (written before `Config.save()` learned to write app-relative paths,
and never re-saved since). On another PC of the same operating system that path is not "foreign"
(`_is_foreign_absolute_db_path` only catches the other OS's path family), so it was left alone:
the app created that folder tree and opened a NEW, EMPTY database there, ignoring the copy sitting
beside it -- a failure that looked like a result.

The rule these tests pin, for an ABSOLUTE configured path (every path below is absolute and native
to whichever OS runs the test, so the rule is exercised the same on the Mac and on Windows):

  1. its file exists                                        -> it is used (the owner's own PC);
  2. it does not, and <app>/data/analysis.db exists         -> the one beside the app, at WARNING;
  3. neither exists and the configured FOLDER is not here   -> the default beside the app, same
                                                               WARNING, and nothing is created at
                                                               the stranger's path;
  4. neither exists but the configured folder is here       -> the configured path, as before.

The cross-OS rule still decides first, and nothing is created on disk for a path that was refused.
"""
import logging
import os
from pathlib import Path

import pytest
import yaml

from laser_trim_analyzer import config as cfg

NOT_HERE = "not on this computer"


def _app(tmp_path, monkeypatch) -> Path:
    """A tmp app directory -- the seam every config test uses -- holding an empty data folder."""
    app = tmp_path / "this_pc" / "LaserTrimAnalyzer"
    (app / "data").mkdir(parents=True)
    monkeypatch.setattr(cfg, "get_app_directory", lambda: app)
    return app


def _config_naming(app: Path, database_path) -> Path:
    """<app>/data/config.yaml whose database.path is exactly this string."""
    config_path = app / "data" / "config.yaml"
    config_path.write_text(yaml.safe_dump({"database": {"path": str(database_path)}}))
    return config_path


def _tree(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _said_not_here(caplog):
    return [r for r in caplog.records
            if r.name == "laser_trim_analyzer.config" and NOT_HERE in r.getMessage()]


def test_case_1_a_configured_database_that_exists_is_the_one_used(tmp_path, monkeypatch, caplog):
    """The owner's own computer: the file his config names is there, so nothing changes -- even
    with another database sitting beside the app."""
    app = _app(tmp_path, monkeypatch)
    (app / "data" / "analysis.db").write_bytes(b"beside the app")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "analysis.db").write_bytes(b"the configured one")
    config_path = _config_naming(app, elsewhere / "analysis.db")
    with caplog.at_level(logging.WARNING):
        loaded = cfg.Config.load(config_path)
    assert loaded.database.path == elsewhere / "analysis.db"
    assert _said_not_here(caplog) == []


@pytest.mark.parametrize("its_folder_is_here", [False, True])
def test_case_2_a_missing_configured_database_gives_way_to_the_one_beside_the_app(
        tmp_path, monkeypatch, caplog, its_folder_is_here):
    """The coworker's computer: config.yaml names the owner's path, which is not here, and the
    copy of the database is beside the app. That copy is opened -- and the log says so, naming
    both paths. Whether the configured FOLDER happens to exist changes nothing."""
    app = _app(tmp_path, monkeypatch)
    beside = app / "data" / "analysis.db"
    beside.write_bytes(b"the copy she was given")
    theirs = tmp_path / "another_pc" / "laser-trim-ai-system" / "data" / "analysis.db"
    if its_folder_is_here:
        theirs.parent.mkdir(parents=True)
    config_path = _config_naming(app, theirs)
    before = _tree(tmp_path)
    with caplog.at_level(logging.WARNING):
        loaded = cfg.Config.load(config_path)
    assert loaded.database.path == beside
    (said,) = _said_not_here(caplog)
    assert said.levelno == logging.WARNING
    assert str(theirs) in said.getMessage() and str(beside) in said.getMessage(), said.getMessage()
    # Nothing was created for the path that was refused -- by the load, or by what the app does
    # next with the path it was given (main() calls ensure_directory()).
    loaded.database.ensure_directory()
    assert _tree(tmp_path) == before
    assert not theirs.exists()


def test_case_3_a_path_from_another_computer_falls_back_to_the_default_beside_the_app(
        tmp_path, monkeypatch, caplog):
    """Neither database exists and the configured folder is not on this computer either: the path
    came from somewhere else. The app uses its own default and never builds a stranger's folder
    tree -- which is exactly what it used to do, before opening an empty database inside it."""
    app = _app(tmp_path, monkeypatch)
    theirs = tmp_path / "another_pc" / "laser-trim-ai-system" / "data" / "analysis.db"
    config_path = _config_naming(app, theirs)
    before = _tree(tmp_path)
    with caplog.at_level(logging.WARNING):
        loaded = cfg.Config.load(config_path)
    assert loaded.database.path == app / "data" / "analysis.db"
    (said,) = _said_not_here(caplog)
    assert said.levelno == logging.WARNING
    assert str(theirs) in said.getMessage()
    assert str(app / "data" / "analysis.db") in said.getMessage(), said.getMessage()
    loaded.database.ensure_directory()
    assert _tree(tmp_path) == before, "something was created on disk"
    assert not (tmp_path / "another_pc").exists(), "a stranger's folder tree was created"


def test_case_4_a_missing_database_in_a_folder_that_exists_is_kept(tmp_path, monkeypatch, caplog):
    """Neither exists, but the configured FOLDER does: a deliberate outside location, or a fresh
    rebuild in place. The configured path stands, as it always did, without a word."""
    app = _app(tmp_path, monkeypatch)
    outside = tmp_path / "a_folder_that_is_here"
    outside.mkdir()
    config_path = _config_naming(app, outside / "analysis.db")
    with caplog.at_level(logging.WARNING):
        loaded = cfg.Config.load(config_path)
    assert loaded.database.path == outside / "analysis.db"
    assert _said_not_here(caplog) == []


@pytest.mark.parametrize("data_folder_is_here", [True, False])
def test_the_apps_own_default_written_absolute_stands_without_a_word(
        tmp_path, monkeypatch, caplog, data_folder_is_here):
    """The owner's config as it is today, read in the folder it was written in, before the
    database exists (a fresh rebuild): the configured path IS the one beside the app, so there is
    nothing to choose between and nothing to warn about -- data folder present or not."""
    app = _app(tmp_path, monkeypatch)
    config_path = _config_naming(app, app / "data" / "analysis.db")
    if not data_folder_is_here:
        config_path = config_path.rename(tmp_path / "config.yaml")
        (app / "data").rmdir()
    with caplog.at_level(logging.WARNING):
        loaded = cfg.Config.load(config_path)
    assert loaded.database.path == app / "data" / "analysis.db"
    assert _said_not_here(caplog) == []


def test_the_cross_os_rule_still_decides_first(tmp_path, monkeypatch, caplog):
    """A path of the OTHER operating system's family is refused by the older rule, with its own
    words, before this one is asked -- and nothing is created for it (the 2026-09-22 junk
    `C:\\dev\\...` database in the repo root)."""
    app = _app(tmp_path, monkeypatch)
    (app / "data" / "analysis.db").write_bytes(b"beside the app")
    foreign = ("/opt/laser-trim-ai-system/data/analysis.db" if os.name == "nt"
               else "C:\\dev\\laser-trim-ai-system\\data\\analysis.db")
    assert cfg._is_foreign_absolute_db_path(foreign)
    config_path = _config_naming(app, foreign)
    before = _tree(tmp_path)
    monkeypatch.chdir(tmp_path)          # where a relative junk name would land
    with caplog.at_level(logging.WARNING):
        loaded = cfg.Config.load(config_path)
    assert loaded.database.path == app / "data" / "analysis.db"
    assert "another operating system" in caplog.text
    assert _said_not_here(caplog) == [], "the new rule spoke for a path the cross-OS rule refused"
    loaded.database.ensure_directory()
    assert _tree(tmp_path) == before, "something was created on disk for a refused path"


def test_the_next_save_writes_the_path_beside_the_app_as_a_relative_one(tmp_path, monkeypatch):
    """Once the copy beside the app has been chosen, saving any setting rewrites config.yaml with
    the app-relative path (existing behaviour): the other computer's path is gone for good."""
    app = _app(tmp_path, monkeypatch)
    (app / "data" / "analysis.db").write_bytes(b"the copy she was given")
    theirs = tmp_path / "another_pc" / "laser-trim-ai-system" / "data" / "analysis.db"
    config_path = _config_naming(app, theirs)
    cfg.Config.load(config_path).save(config_path)
    stored = yaml.safe_load(config_path.read_text())["database"]["path"]
    assert stored == "data/analysis.db", stored
    assert cfg.Config.load(config_path).database.path == app / "data" / "analysis.db"


def test_a_relative_path_is_still_resolved_against_the_app_folder(tmp_path, monkeypatch, caplog):
    """The rule is about ABSOLUTE paths only: a relative one means the app folder, as before."""
    app = _app(tmp_path, monkeypatch)
    config_path = _config_naming(app, "data/analysis.db")
    with caplog.at_level(logging.WARNING):
        loaded = cfg.Config.load(config_path)
    assert loaded.database.path == app / "data" / "analysis.db"
    assert _said_not_here(caplog) == []


# ---- the packaged build: beside the exe ALWAYS wins ---------------------------------------------
# James, 2026-09-30: "i dont want her reading my db". The four cases above keep his own laptop
# unchanged (case 1 follows a configured path that EXISTS) -- which is also how a packaged build
# would reach his database: on his own laptop, where C:\dev\...\analysis.db is there, or on her PC
# if his config ever named a share she can reach. A packaged build therefore never follows the
# configured path at all.

PACKAGED = "packaged build"


def _said_packaged(caplog):
    return [r for r in caplog.records
            if r.name == "laser_trim_analyzer.config" and PACKAGED in r.getMessage()]


def test_a_packaged_build_opens_only_the_database_beside_its_exe(tmp_path, monkeypatch, caplog):
    """The configured database EXISTS on this computer (case 1 would use it) -- the packaged
    build still opens the one beside the exe, and says which it ignored."""
    app = _app(tmp_path, monkeypatch)
    (app / "data" / "analysis.db").write_bytes(b"beside the exe")
    elsewhere = tmp_path / "the_owners_folder"
    elsewhere.mkdir()
    (elsewhere / "analysis.db").write_bytes(b"the owner's own database")
    config_path = _config_naming(app, elsewhere / "analysis.db")
    monkeypatch.setattr(cfg.sys, "frozen", True, raising=False)
    with caplog.at_level(logging.WARNING):
        loaded = cfg.Config.load(config_path)
    assert loaded.database.path == app / "data" / "analysis.db"
    said = _said_packaged(caplog)
    assert len(said) == 1 and str(elsewhere / "analysis.db") in said[0].getMessage()
    assert (elsewhere / "analysis.db").read_bytes() == b"the owner's own database"   # never touched


def test_a_packaged_build_ignores_a_relative_path_that_leaves_its_folder(tmp_path, monkeypatch,
                                                                        caplog):
    """A relative path is resolved against the app folder -- and can still climb out of it."""
    app = _app(tmp_path, monkeypatch)
    outside = app.parent / "outside"
    outside.mkdir()
    (outside / "analysis.db").write_bytes(b"outside the app folder")
    config_path = _config_naming(app, "../outside/analysis.db")
    monkeypatch.setattr(cfg.sys, "frozen", True, raising=False)
    with caplog.at_level(logging.WARNING):
        loaded = cfg.Config.load(config_path)
    assert loaded.database.path == app / "data" / "analysis.db"
    assert len(_said_packaged(caplog)) == 1


def test_a_packaged_build_says_nothing_when_the_config_already_means_beside_the_exe(
        tmp_path, monkeypatch, caplog):
    app = _app(tmp_path, monkeypatch)
    (app / "data" / "analysis.db").write_bytes(b"beside the exe")
    monkeypatch.setattr(cfg.sys, "frozen", True, raising=False)
    for named in ("data/analysis.db", app / "data" / "analysis.db"):
        config_path = _config_naming(app, named)
        with caplog.at_level(logging.WARNING):
            loaded = cfg.Config.load(config_path)
        assert loaded.database.path == app / "data" / "analysis.db"
    assert _said_packaged(caplog) == []


def test_from_source_the_owners_configured_database_is_still_followed(tmp_path, monkeypatch):
    """Not packaged (run_v6.bat): case 1 is unchanged -- his laptop opens what his config names."""
    app = _app(tmp_path, monkeypatch)
    (app / "data" / "analysis.db").write_bytes(b"beside the app")
    elsewhere = tmp_path / "the_owners_folder"
    elsewhere.mkdir()
    (elsewhere / "analysis.db").write_bytes(b"the owner's own database")
    config_path = _config_naming(app, elsewhere / "analysis.db")
    monkeypatch.delattr(cfg.sys, "frozen", raising=False)
    assert cfg.Config.load(config_path).database.path == elsewhere / "analysis.db"
