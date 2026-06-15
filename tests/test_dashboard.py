"""Dashboard (Production Health) + helpers. Fixtures in tests/conftest.py."""
from datetime import datetime, timedelta

import pytest

from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAR, FinalTestResult as DBFT, SystemType, StatusType)


_SEQ = [0]


def _uid() -> int:
    """Monotonic id so seeded rows never collide on the unique constraints."""
    _SEQ[0] += 1
    return _SEQ[0]


def _add_ar(s, model, status, when):
    u = _uid()
    s.add(DBAR(filename=f"{model}-{status.name}-{u}.xls",
               file_path="/f/x.xls", file_hash=f"har{u}",
               model=model, serial=f"sn{u}", system=SystemType.A, file_date=when, timestamp=when,
               overall_status=status, has_multi_tracks=False, processing_time=0.1))


def _add_ft(s, model, status, when):
    u = _uid()
    s.add(DBFT(filename=f"ft-{model}-{status.name}-{u}.xls", file_path="/f/ft.xls",
               file_hash=f"hft{u}", model=model, serial=f"sn{u}",
               file_date=when, test_date=when, timestamp=when, overall_status=status))


def test_compute_yield_empty(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    y = compute_yield(DatabaseManager(tmp_path / "e.db"), DBAR, None)
    assert y["total"] == 0 and y["pass_rate"] is None and y["trend"] == []


def test_compute_yield_buckets_and_rate(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "y.db")
    now = datetime.now()
    with db.session() as s:
        for _ in range(3):
            _add_ar(s, "M", StatusType.PASS, now)
        _add_ar(s, "M", StatusType.WARNING, now)
        _add_ar(s, "M", StatusType.FAIL, now)
        _add_ar(s, "M", StatusType.UNTRIMMED, now)   # excluded from rate
        s.commit()
    y = compute_yield(db, DBAR, None)
    assert (y["passed"], y["warnings"], y["failed"], y["untrimmed"]) == (3, 1, 1, 1)
    assert y["gradeable"] == 5
    assert y["pass_rate"] == pytest.approx(60.0)     # 3 / (3+1+1)


def test_compute_yield_windowed(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "w.db")
    now = datetime.now()
    with db.session() as s:
        _add_ar(s, "M", StatusType.PASS, now)
        _add_ar(s, "M", StatusType.PASS, now - timedelta(days=200))   # outside 90d
        s.commit()
    assert compute_yield(db, DBAR, now - timedelta(days=90))["total"] == 1


def test_compute_yield_on_final_test(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "ft.db")
    now = datetime.now()
    with db.session() as s:
        _add_ft(s, "M", StatusType.PASS, now)
        _add_ft(s, "M", StatusType.FAIL, now)
        s.commit()
    y = compute_yield(db, DBFT, None)
    assert y["passed"] == 1 and y["failed"] == 1 and y["pass_rate"] == pytest.approx(50.0)


def test_worst_models_ranks_and_min_units(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import worst_models_by_yield
    db = DatabaseManager(tmp_path / "wm.db")
    now = datetime.now()
    with db.session() as s:
        for _ in range(5):
            _add_ar(s, "GOOD", StatusType.PASS, now)
        for _ in range(3):
            _add_ar(s, "BAD", StatusType.PASS, now)
        for _ in range(2):
            _add_ar(s, "BAD", StatusType.FAIL, now)
        _add_ar(s, "TINY", StatusType.FAIL, now)        # below min_units, excluded
        s.commit()
    rows, total = worst_models_by_yield(db, None, min_units=5, limit=10)
    assert [r["model"] for r in rows] == ["BAD", "GOOD"]   # worst first
    assert total == 2                                       # TINY excluded by min_units
    assert rows[0]["units"] == 5 and rows[0]["trim_rate"] == pytest.approx(60.0)


def test_worst_models_cap_disclosed(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import worst_models_by_yield
    db = DatabaseManager(tmp_path / "cap.db")
    now = datetime.now()
    with db.session() as s:
        for i in range(12):
            for _ in range(5):
                _add_ar(s, f"M{i:02d}", StatusType.PASS, now)
        s.commit()
    rows, total = worst_models_by_yield(db, None, min_units=5, limit=10)
    assert len(rows) == 10 and total == 12


def test_worst_models_joins_ft_rate(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import worst_models_by_yield
    db = DatabaseManager(tmp_path / "j.db")
    now = datetime.now()
    with db.session() as s:
        for _ in range(5):
            _add_ar(s, "M", StatusType.PASS, now)
        _add_ft(s, "M", StatusType.PASS, now)
        _add_ft(s, "M", StatusType.FAIL, now)     # FT 50%
        s.commit()
    rows, _ = worst_models_by_yield(db, None, min_units=5, limit=10)
    assert rows[0]["ft_rate"] == pytest.approx(50.0)


def test_mini_trend_chart_set_points_no_crash(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.mini_trend_chart import MiniTrendChart
    c = MiniTrendChart(tk_root, theme=ThemeManager())
    c.set_points([("2026-05-01", 90.0), ("2026-05-02", 80.0), ("2026-05-03", 95.0)])


def test_mini_trend_chart_empty_no_crash(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.mini_trend_chart import MiniTrendChart
    MiniTrendChart(tk_root, theme=ThemeManager()).set_points([])


def _labels_text(widget):
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels_text(c))
    return out


def test_yield_panel_renders_rate_and_counts(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.yield_panel import YieldPanel
    p = YieldPanel(tk_root, theme=ThemeManager(), title="Trim analysis yield")
    p.set_yield({"passed": 3, "warnings": 1, "failed": 1, "errors": 0, "untrimmed": 1,
                 "gradeable": 5, "total": 6, "pass_rate": 60.0, "trend": []},
                total_label="6 units")
    txt = " | ".join(_labels_text(p))
    assert "Trim analysis yield" in txt
    assert "60" in txt                # headline %
    assert "3" in txt and "1" in txt  # pass / warn-fail counts
    assert "6 units" in txt


def test_yield_panel_empty_state(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.yield_panel import YieldPanel
    p = YieldPanel(tk_root, theme=ThemeManager(), title="Final-test yield")
    p.set_yield({"passed": 0, "warnings": 0, "failed": 0, "errors": 0, "untrimmed": 0,
                 "gradeable": 0, "total": 0, "pass_rate": None, "trend": []},
                total_label="0 matched")
    assert "—" in " ".join(_labels_text(p))   # no fabricated 0%


def test_worst_models_list_rows_and_click(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
    got = []
    w = WorstModelsList(tk_root, theme=ThemeManager(), on_row_click=got.append)
    w.set_rows([{"model": "BAD", "units": 5, "trim_rate": 60.0, "ft_rate": 48.0},
                {"model": "OK", "units": 9, "trim_rate": 95.0, "ft_rate": None}], total=2)
    assert len(w._rows) == 2
    w._rows[0]._on_click()
    assert got == ["BAD"]


def test_worst_models_list_discloses_cap(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
    w = WorstModelsList(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    w.set_rows([{"model": f"M{i}", "units": 5, "trim_rate": 50.0, "ft_rate": None}
                for i in range(10)], total=25)
    assert "10 of 25" in w._cap.cget("text")


def test_worst_models_list_empty(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
    w = WorstModelsList(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    w.set_rows([], total=0)
    assert w._rows == []


def test_dashboard_is_landing(make_app):
    app = make_app()
    assert app.page_container.current_page == "dashboard"
    assert app.page_container.get_page("dashboard") is not None


def test_dashboard_reload_now_populates(make_app):
    app = make_app()
    now = datetime.now()
    with app.db.session() as s:
        for _ in range(5):
            _add_ar(s, "DASH", StatusType.PASS, now)
        _add_ar(s, "DASH", StatusType.FAIL, now)
        s.commit()
    page = app.page_container.get_page("dashboard")
    page.reload_now()
    # trim panel shows a rate; worst-models has DASH (5 pass + 1 fail = 6 gradeable >= 5)
    assert any("83" in x or "%" in x for x in _labels_text(page._trim_panel))
    assert any(r.row["model"] == "DASH" for r in page._worst._rows)


def test_dashboard_row_click_routes_to_model(make_app):
    app = make_app()
    page = app.page_container.get_page("dashboard")
    page._on_model_click("ROUTED")
    assert app.page_container.current_page == "model"
    # The real Model page consumes the route on show and lands on the model.
    assert app.page_container.get_page("model")._current_model == "ROUTED"
    assert app._model_route is None
