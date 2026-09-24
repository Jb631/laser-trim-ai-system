"""A data/ folder carried between Windows and the Mac must not carry a path that only
exists on one of them (2026-09-22: a junk `C:\\dev\\...\\analysis.db` appeared in the
repo root on the Mac)."""
import os

import pytest

from laser_trim_analyzer import config as cfg


def test_a_path_inside_the_app_folder_is_saved_relative(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "get_app_directory", lambda: tmp_path)
    c = cfg.Config()
    c.database.path = tmp_path / "data" / "analysis.db"
    c.save(tmp_path / "data" / "config.yaml")
    text = (tmp_path / "data" / "config.yaml").read_text()
    assert str(tmp_path) not in text and "data/analysis.db" in text.replace("\\", "/")
    assert cfg.Config.load(tmp_path / "data" / "config.yaml").database.path == tmp_path / "data" / "analysis.db"


def test_a_windows_path_read_on_another_os_falls_back_to_the_default(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cfg, "get_app_directory", lambda: tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "config.yaml").write_text(
        "database:\n  path: 'C:\\dev\\laser-trim-ai-system\\data\\analysis.db'\n")
    loaded = cfg.Config.load(tmp_path / "data" / "config.yaml")
    if os.name != "nt":
        assert loaded.database.path == tmp_path / "data" / "analysis.db"
        assert "another" in caplog.text.lower() or "windows" in caplog.text.lower()


@pytest.mark.skipif(os.name != "nt", reason="mirror case only applies on Windows")
def test_a_posix_path_read_on_windows_falls_back_to_the_default(tmp_path, monkeypatch, caplog):
    """The mirror of the Windows-path test above: a POSIX absolute path (as would be
    carried from the Mac inside data/) read under os.name == 'nt' is just as foreign,
    and must fall back the same way."""
    monkeypatch.setattr(cfg, "get_app_directory", lambda: tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "config.yaml").write_text(
        "database:\n  path: '/opt/laser-trim-ai-system/data/analysis.db'\n")
    loaded = cfg.Config.load(tmp_path / "data" / "config.yaml")
    assert loaded.database.path == tmp_path / "data" / "analysis.db"
    assert "another" in caplog.text.lower() or "posix" in caplog.text.lower() \
        or "windows" in caplog.text.lower()


def test_a_native_absolute_path_outside_the_app_folder_round_trips_unchanged(tmp_path, monkeypatch):
    """The cost the design accepts: a user who deliberately keeps the database
    outside the app folder, on THIS machine's own OS, gets that path back exactly --
    they just have to set it again on the other machine.

    `outside_dir` is a SIBLING of `tmp_path`, not a path under it (conftest's
    autouse DB guard already claims a directory named "app_root" under
    `tmp_path`, and the point of this test is a location the app folder does
    not contain), named from tmp_path's own unique suffix so parallel test
    runs cannot collide.
    """
    monkeypatch.setattr(cfg, "get_app_directory", lambda: tmp_path)
    outside_dir = tmp_path.parent / (tmp_path.name + "_outside")
    outside_dir.mkdir(exist_ok=True)
    c = cfg.Config()
    c.database.path = outside_dir / "analysis.db"
    config_path = tmp_path / "data" / "config.yaml"
    c.save(config_path)
    loaded = cfg.Config.load(config_path)
    assert loaded.database.path == outside_dir / "analysis.db"
