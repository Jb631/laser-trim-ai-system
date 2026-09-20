"""scripts/_db_guard.is_production_db, tested as a pure function against a decoy repo.

No test here may name the real data/analysis.db: the guard IS the refusal that protects it,
so proving it works must never depend on a test that already points at the real file. Every
case below builds its own decoy repo/data/analysis.db under tmp_path and calls the loaded
module directly -- no subprocess, no real path anywhere in this file.
"""
import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD_PATH = REPO / "scripts" / "_db_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("_db_guard_under_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def guard():
    return _load_guard()


@pytest.fixture
def decoy(tmp_path):
    """A decoy checkout: tmp_path/repo/data/analysis.db, a few real bytes."""
    repo = tmp_path / "repo"
    data_dir = repo / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "analysis.db").write_bytes(b"decoy production bytes")
    return repo


def _case_insensitive(decoy: Path) -> bool:
    return (decoy / "data" / "ANALYSIS.DB").exists()


def test_the_decoy_production_path_itself_is_production(guard, decoy):
    prod = decoy / "data" / "analysis.db"
    assert guard.is_production_db(prod, decoy) is True, \
        "the production path itself must be refused"


def test_a_hard_link_to_the_production_file_is_production(guard, decoy, tmp_path):
    prod = decoy / "data" / "analysis.db"
    alias = tmp_path / "alias_hardlink.db"
    try:
        os.link(prod, alias)
    except OSError as exc:
        pytest.skip(f"cannot hard-link on this filesystem: {exc}")
    assert guard.is_production_db(alias, decoy) is True, \
        "a hard link to the production file shares its identity and must be refused"


def test_a_symlink_to_the_production_file_is_production(guard, decoy, tmp_path):
    prod = decoy / "data" / "analysis.db"
    alias = tmp_path / "alias_symlink.db"
    try:
        alias.symlink_to(prod)
    except OSError as exc:
        pytest.skip(f"cannot symlink on this filesystem: {exc}")
    assert guard.is_production_db(alias, decoy) is True, \
        "a symlink to the production file resolves to the same identity and must be refused"


def test_a_case_variant_of_the_path_is_production_on_a_case_insensitive_volume(guard, decoy):
    if not _case_insensitive(decoy):
        pytest.skip("this filesystem is case-sensitive -- data/Analysis.db is a different file here")
    variant = decoy / "data" / "Analysis.db"
    assert guard.is_production_db(variant, decoy) is True, \
        "on a case-insensitive volume, data/Analysis.db IS data/analysis.db"


def test_the_case_variant_is_caught_by_identity_alone_even_with_by_name_false(guard, decoy):
    """The whole point of comparing by identity: with the name rule turned off, only
    os.path.samefile is left to catch the case variant on a case-insensitive volume --
    and it must."""
    if not _case_insensitive(decoy):
        pytest.skip("this filesystem is case-sensitive -- data/Analysis.db is a different file here")
    variant = decoy / "data" / "Analysis.db"
    assert guard.is_production_db(variant, decoy, by_name=False) is True, \
        "by_name=False must still be caught by os.path.samefile on a case-insensitive volume"


def test_a_nonexistent_analysis_db_is_refused_by_name(guard, decoy, tmp_path):
    ghost = tmp_path / "analysis.db"
    assert guard.is_production_db(ghost, decoy, by_name=True) is True, \
        "any path literally named analysis.db must be refused by name, even if it doesn't exist"


def test_a_nonexistent_mixed_case_name_is_refused_by_name(guard, decoy, tmp_path):
    """The path does not exist, so os.path.samefile never runs (path.exists() is False) --
    only the by-name casefold comparison can catch this one. Unlike the case-variant test
    above, this does not depend on the filesystem being case-insensitive."""
    ghost = tmp_path / "Analysis.db"
    assert guard.is_production_db(ghost, decoy, by_name=True) is True, \
        "a mixed-case name must be refused by casefold, regardless of filesystem case-sensitivity"


def test_a_nonexistent_analysis_db_is_not_refused_when_by_name_is_false(guard, decoy, tmp_path):
    ghost = tmp_path / "analysis.db"
    assert guard.is_production_db(ghost, decoy, by_name=False) is False, \
        "with by_name=False, a nonexistent path cannot be proven to be the production file"


def test_an_ordinary_copy_with_different_bytes_is_not_production(guard, decoy, tmp_path):
    copy = tmp_path / "qa_copy.db"
    copy.write_bytes(b"different bytes entirely, not the decoy production file")
    assert guard.is_production_db(copy, decoy) is False, \
        "an ordinary copy under another name, with different content, must be accepted"


def test_a_samefile_oserror_is_refused_because_identity_cannot_be_proven(guard, decoy, tmp_path, monkeypatch):
    copy = tmp_path / "qa_copy.db"
    copy.write_bytes(b"different bytes entirely, not the decoy production file")

    def boom(_a, _b):
        raise OSError("simulated I/O error -- identity cannot be established")

    monkeypatch.setattr(guard.os.path, "samefile", boom)
    assert guard.is_production_db(copy, decoy) is True, \
        "when samefile() cannot prove the path is NOT production, the guard must refuse"
