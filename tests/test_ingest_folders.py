"""Ingest folder list — config storage + the Settings section that edits it.

Spec: docs/superpowers/specs/2026-08-29-app-shape-investigate-design.md §1
("Folders are configured once (add/remove/reorder in Settings)") and §3
("SETTINGS — unchanged (gains only the ingest folder list)").

The list is ORDERED on purpose: James runs the laser folders and then the
Final Test folder, and HOME's one-click ingest walks them in exactly that
order. So order preservation is a tested contract, not an accident of the
storage format.
"""
from pathlib import Path

import pytest

from laser_trim_analyzer.config import (
    Config,
    IngestConfig,
    ingest_folder_problem,
    missing_ingest_folders,
    normalize_ingest_path,
)


# ---- Task 1: normalization + the ordered list ------------------------------

def test_ingest_folders_default_is_empty():
    assert Config().ingest.folders == []


def test_normalize_strips_whitespace_and_trailing_separators():
    assert normalize_ingest_path("  /data/laser/  ") == "/data/laser"
    assert normalize_ingest_path("/data/laser") == "/data/laser"
    # A root is not a trailing separator to be eaten.
    assert normalize_ingest_path("/") == "/"
    # UNC shares are the real-world case (\\192.168.66.9\...): backslashes are
    # separators too, and the trailing one is cosmetic.
    assert normalize_ingest_path("\\\\192.168.66.9\\Laser\\") == "\\\\192.168.66.9\\Laser"
    assert normalize_ingest_path("") == ""
    assert normalize_ingest_path("   ") == ""


def test_add_appends_in_order():
    cfg = IngestConfig()
    assert cfg.add("/a") is True
    assert cfg.add("/b") is True
    assert cfg.add("/c") is True
    assert cfg.folders == ["/a", "/b", "/c"]


def test_add_rejects_duplicate_including_trailing_slash_variant():
    cfg = IngestConfig()
    assert cfg.add("/a") is True
    assert cfg.add("/a") is False
    assert cfg.add("/a/") is False          # same folder, cosmetic difference
    assert cfg.add("  /a  ") is False
    assert cfg.folders == ["/a"]


def test_add_rejects_empty():
    cfg = IngestConfig()
    assert cfg.add("") is False
    assert cfg.add("   ") is False
    assert cfg.folders == []


def test_remove_by_path_and_unknown_is_noop():
    cfg = IngestConfig(folders=["/a", "/b", "/c"])
    assert cfg.remove("/b/") is True         # normalized match
    assert cfg.folders == ["/a", "/c"]
    assert cfg.remove("/nope") is False
    assert cfg.folders == ["/a", "/c"]


def test_move_reorders_and_respects_bounds():
    cfg = IngestConfig(folders=["/a", "/b", "/c"])
    assert cfg.move(2, -1) is True
    assert cfg.folders == ["/a", "/c", "/b"]
    assert cfg.move(0, +1) is True
    assert cfg.folders == ["/c", "/a", "/b"]
    # Off either end: refuse, and leave the list exactly as it was.
    assert cfg.move(0, -1) is False
    assert cfg.move(2, +1) is False
    assert cfg.move(7, -1) is False
    assert cfg.folders == ["/c", "/a", "/b"]


def test_index_of_normalizes():
    cfg = IngestConfig(folders=["/a", "/b"])
    assert cfg.index_of("/b/") == 1
    assert cfg.index_of("/zzz") is None


# ---- Task 2: config round-trip --------------------------------------------

def test_round_trip_preserves_order(tmp_path):
    p = tmp_path / "config.yaml"
    cfg = Config()
    for f in ("/laser/one", "/laser/two", "/final_test"):
        cfg.ingest.add(f)
    cfg.save(p)
    again = Config.load(p)
    assert again.ingest.folders == ["/laser/one", "/laser/two", "/final_test"]


def test_round_trip_empty_list(tmp_path):
    """An empty list must survive as an empty list — HOME's empty state keys
    off it, so 'missing key' and 'no folders' have to look the same."""
    p = tmp_path / "config.yaml"
    Config().save(p)
    assert Config.load(p).ingest.folders == []


def test_load_tolerates_missing_and_junk_section(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("gui:\n  theme: dark\n")
    assert Config.load(p).ingest.folders == []
    p.write_text("ingest:\n  folders: 'not a list'\n")
    assert Config.load(p).ingest.folders == []


def test_round_trip_of_unc_path(tmp_path):
    """Backslash-heavy UNC paths must come back byte-identical — a YAML
    round-trip that mangles \\\\192.168.66.9\\... points the batch at nothing."""
    p = tmp_path / "config.yaml"
    cfg = Config()
    unc = "\\\\192.168.66.9\\Public\\LaserTrim"
    cfg.ingest.add(unc)
    cfg.save(p)
    assert Config.load(p).ingest.folders == [unc]


# ---- Task 3: missing / unreadable folders are REPORTED, never skipped ------

def test_folder_problem_none_for_a_real_directory(tmp_path):
    assert ingest_folder_problem(str(tmp_path)) is None


def test_folder_problem_names_a_missing_folder(tmp_path):
    msg = ingest_folder_problem(str(tmp_path / "gone"))
    assert msg and "not found" in msg.lower()


def test_folder_problem_flags_a_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    msg = ingest_folder_problem(str(f))
    assert msg and "folder" in msg.lower()


def test_folder_problem_flags_empty_path():
    assert ingest_folder_problem("") is not None


def test_missing_ingest_folders_reports_only_the_broken_ones(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    bad = str(tmp_path / "offline_share")
    out = missing_ingest_folders([str(good), bad])
    assert [p for p, _ in out] == [bad]
    assert out[0][1]                          # carries a reason, not just a flag


def test_missing_ingest_folders_empty_list_is_empty():
    assert missing_ingest_folders([]) == []


# ---- Task 4: the Settings section -----------------------------------------

def _section(app, tk_root):
    from laser_trim_analyzer.gui.v6.sections.ingest_folders import (
        build_ingest_folders_section,
    )
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    import customtkinter as ctk
    parent = ctk.CTkFrame(tk_root)
    return build_ingest_folders_section(parent, theme=ThemeManager(), app=app)


class _StubApp:
    def __init__(self, config, tmp_path):
        self.config = config
        self._saved_to = tmp_path / "config.yaml"

    def save(self):
        self.config.save(self._saved_to)


def test_settings_page_has_an_ingest_folders_card(make_app):
    app = make_app()
    page = app.page_container.get_page("settings")
    titles = [c._title.cget("text") for c in page._cards]
    assert any("Ingest Folders" in t for t in titles), titles


def test_section_add_remove_reorder_writes_through_to_config(tk_root, tmp_path,
                                                             monkeypatch):
    cfg = Config()
    cfg_path = tmp_path / "config.yaml"
    monkeypatch.setattr(Config, "save", lambda self, path=None: None)

    class _App:
        config = cfg
    sec = _section(_App(), tk_root)

    assert sec.add_folder("/laser/one") is None          # None == accepted
    assert sec.add_folder("/laser/two") is None
    assert sec.paths() == ["/laser/one", "/laser/two"]
    # Duplicate comes back as a message the user can read, and changes nothing.
    msg = sec.add_folder("/laser/one/")
    assert msg and "already" in msg.lower()
    assert sec.paths() == ["/laser/one", "/laser/two"]
    sec.move(1, -1)
    assert sec.paths() == ["/laser/two", "/laser/one"]
    sec.remove(0)
    assert sec.paths() == ["/laser/one"]
    assert cfg.ingest.folders == ["/laser/one"]


def test_section_reports_a_missing_folder_without_refusing_it(tk_root, tmp_path,
                                                              monkeypatch):
    """An offline share is still a configured folder. Adding it must WORK and
    say it is unreachable — silently dropping it is how a batch quietly
    processes two of three folders."""
    monkeypatch.setattr(Config, "save", lambda self, path=None: None)

    class _App:
        config = Config()
    sec = _section(_App(), tk_root)
    gone = str(tmp_path / "offline")
    assert sec.add_folder(gone) is None
    assert sec.paths() == [gone]
    status = sec.status_text()
    assert "1 folder" in status
    assert "unreachable" in status.lower() or "not found" in status.lower()


def test_section_empty_state_says_so(tk_root, monkeypatch):
    monkeypatch.setattr(Config, "save", lambda self, path=None: None)

    class _App:
        config = Config()
    sec = _section(_App(), tk_root)
    assert sec.paths() == []
    assert "no folders" in sec.status_text().lower()
