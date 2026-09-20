"""Settings — Backlog section (E1, 2026-09-20). Fixtures in tests/conftest.py.

Covers the 7 points task-E1-brief.md names for this file, in the same order.
Synthetic rows only — never data/analysis.db, never Work Files/.
"""
from datetime import datetime

import pandas as pd
import pytest

from laser_trim_analyzer.core.backlog import BacklogFormatError
from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType, StatusType
from laser_trim_analyzer.gui.v6.sections.backlog import apply_backlog, build_backlog_section


def _backlog_df(rows):
    return pd.DataFrame(rows, columns=["Customer Name", "PO Number", "Order Date", "Need Date",
                                       "Item ID", "Balance", "Unit Price"])


def _seed_known_models(app, models):
    """Make each of `models` a "known model" (present in analysis_results), the
    same way the real app would after processing at least one file for it."""
    with app.db.session() as s:
        for i, model in enumerate(models):
            s.add(DBAR(filename=f"{model}-{i}.xls", file_path=f"/f/{model}-{i}.xls",
                       file_hash=f"h-{model}-{i}", model=model, serial=f"sn{i}",
                       system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                       overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.commit()


# ---- 1. apply_backlog sets fields, rebuilds mps_models, saves ------------------

def test_apply_backlog_sets_config_fields_and_saves(make_app, monkeypatch):
    app = make_app()
    _seed_known_models(app, ["8506", "8232-1"])
    saved = []
    monkeypatch.setattr(app.config, "save", lambda: saved.append(True))

    df = _backlog_df([
        ["ACME", "PO1", "2026-01-01", "2026-10-01", "8506", 5, 100.0],
        ["BOLT", "PO2", "2026-01-02", "2026-11-01", "8232-1", 3, 50.0],
    ])
    apply_backlog(app, df, "Backlog test.xls")

    am = app.config.active_models
    assert am.backlog_models == ["8232-1", "8506"]
    assert am.backlog_open_qty == {"8232-1": 3, "8506": 5}
    assert am.model_prices == {"8232-1": 50.0, "8506": 100.0}
    assert am.backlog_source.startswith("Backlog test.xls")
    assert am.mps_models == sorted(set(am.backlog_models) | set(am.pinned_models))
    assert saved == [True]


# ---- 2. a SECOND upload REPLACES backlog_models but KEEPS the old price -------

def test_second_upload_replaces_models_but_keeps_old_price(make_app):
    app = make_app()
    _seed_known_models(app, ["8506", "8232-1"])

    df1 = _backlog_df([
        ["ACME", "PO1", "2026-01-01", "2026-10-01", "8506", 5, 100.0],
        ["BOLT", "PO2", "2026-01-02", "2026-11-01", "8232-1", 3, 50.0],
    ])
    apply_backlog(app, df1, "week1.xls")

    df2 = _backlog_df([
        ["ACME", "PO3", "2026-02-01", "2026-10-05", "8506", 7, 120.0],
    ])
    apply_backlog(app, df2, "week2.xls")

    am = app.config.active_models
    assert am.backlog_models == ["8506"]                          # 8232-1 dropped from THIS backlog
    assert am.model_prices == {"8506": 120.0, "8232-1": 50.0}     # but its price is kept
    assert am.mps_models == ["8506"]                              # not pinned, so it drops out


# ---- 3. a pinned model stays in mps_models across an upload that lacks it -----

def test_pinned_model_survives_an_upload_without_it(make_app):
    app = make_app()
    _seed_known_models(app, ["8506"])
    app.config.active_models.pinned_models = ["LEGACY-1"]

    df = _backlog_df([["ACME", "PO1", "2026-01-01", "2026-10-01", "8506", 5, 100.0]])
    apply_backlog(app, df, "week1.xls")

    am = app.config.active_models
    assert am.mps_models == sorted({"8506", "LEGACY-1"})
    assert "LEGACY-1" in am.mps_models


# ---- 4. one-time migration: an old hand-pinned mps_models moves to pinned_models --

def test_migration_moves_old_mps_list_into_pinned_models(make_app):
    import customtkinter as ctk
    app = make_app()
    app.config.active_models.mps_models = ["X"]
    assert app.config.active_models.pinned_models == []   # nothing migrated yet
    assert app.config.active_models.backlog_models == []

    build_backlog_section(ctk.CTkFrame(app), theme=app.theme, app=app)

    assert app.config.active_models.pinned_models == ["X"]


# ---- 5. a BacklogFormatError changes nothing and saves nothing ----------------

def test_format_error_changes_nothing_and_saves_nothing(make_app, monkeypatch):
    app = make_app()
    _seed_known_models(app, ["8506"])
    am = app.config.active_models
    am.backlog_models = ["OLD"]
    am.model_prices = {"OLD": 9.0}
    am.backlog_open_qty = {"OLD": 2}
    am.backlog_source = "old.xls"
    before = (list(am.backlog_models), dict(am.model_prices), dict(am.backlog_open_qty),
             am.backlog_source, list(am.mps_models))
    saved = []
    monkeypatch.setattr(app.config, "save", lambda: saved.append(True))

    bad_df = pd.DataFrame({"Item ID": ["8506"], "foo": [1]})   # no Balance / Unit Price columns
    with pytest.raises(BacklogFormatError):
        apply_backlog(app, bad_df, "bad.xls")

    after = (list(am.backlog_models), dict(am.model_prices), dict(am.backlog_open_qty),
            am.backlog_source, list(am.mps_models))
    assert after == before
    assert saved == []


# ---- 6. the four new fields round-trip; an old config file loads with defaults --

def test_new_fields_round_trip_and_old_config_loads_with_defaults(tmp_path):
    from laser_trim_analyzer.config import Config

    cfg = Config()
    cfg.active_models.pinned_models = ["P1"]
    cfg.active_models.backlog_models = ["B1", "B2"]
    cfg.active_models.backlog_open_qty = {"B1": 3, "B2": 5}
    cfg.active_models.backlog_source = "src.xls · uploaded 2026-09-20"
    path = tmp_path / "config.yaml"
    cfg.save(path)

    reloaded = Config.load(path)
    am = reloaded.active_models
    assert am.pinned_models == ["P1"]
    assert am.backlog_models == ["B1", "B2"]
    assert am.backlog_open_qty == {"B1": 3, "B2": 5}
    assert am.backlog_source == "src.xls · uploaded 2026-09-20"

    # A config.yaml written before these four fields existed must still load fine.
    old_path = tmp_path / "old_config.yaml"
    old_path.write_text(
        "active_models:\n"
        "  mps_models:\n"
        "    - OLDMODEL\n"
        "  recent_days: 45\n"
        "  model_prices:\n"
        "    OLDMODEL: 12.5\n"
        "  cost_ratio: 0.4\n"
    )
    old_cfg = Config.load(old_path)
    old_am = old_cfg.active_models
    assert old_am.mps_models == ["OLDMODEL"]          # pre-existing fields still load
    assert old_am.recent_days == 45
    assert old_am.pinned_models == []                 # new fields default sanely
    assert old_am.backlog_models == []
    assert old_am.backlog_open_qty == {}
    assert old_am.backlog_source == ""


# ---- 7. Settings page shows the new title, not the two old ones ---------------

def test_settings_page_shows_backlog_section_not_the_old_two(make_app):
    app = make_app()
    page = app.page_container.get_page("settings")
    titles = [c._title.cget("text") for c in page._cards]
    assert "Backlog — active models and pricing" in titles
    assert not any("Active Models (MPS" in t for t in titles)
    assert "Pricing" not in titles
