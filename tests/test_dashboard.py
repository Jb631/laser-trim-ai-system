"""Dashboard (Production Health) + helpers. Fixtures in tests/conftest.py."""
import logging
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


def test_compute_yield_excludes_future_dated(tmp_path):
    """Regression (2026-07-08): one FT file named `..._12-18-2026_...` put a
    future-dated record in the DB, stretching the dashboard sparkline to Dec
    2026 and making its last-day value a fake 100%. Future rows (beyond a 1-day
    clock-skew allowance) must be excluded from counts AND trend, and counted
    in `future_dated` so the panel can disclose them."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_yield
    db = DatabaseManager(tmp_path / "fd.db")
    now = datetime.now()
    with db.session() as s:
        _add_ft(s, "M", StatusType.PASS, now)
        _add_ft(s, "M", StatusType.FAIL, now)
        _add_ft(s, "M", StatusType.PASS, now + timedelta(days=163))  # mislabeled file
        s.commit()
    y = compute_yield(db, DBFT, None)
    assert y["future_dated"] == 1
    assert (y["passed"], y["failed"]) == (1, 1)          # future PASS not counted
    assert y["pass_rate"] == pytest.approx(50.0)          # not inflated to 66.7
    assert all(p["date"] <= now.strftime("%Y-%m-%d") for p in y["trend"])


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


def test_mini_trend_downsample_bounds_point_count():
    """A long per-day trend (e.g. the 'All' window) must bin down to <=_MAX_POINTS so
    the sparkline stays legible instead of rendering thousands of points as a smear."""
    from laser_trim_analyzer.gui.v6.widgets.mini_trend_chart import MiniTrendChart
    short = [float(i) for i in range(10)]
    assert MiniTrendChart._downsample(short, 48) == short          # below cap: unchanged
    long = [float(i % 100) for i in range(3397)]
    out = MiniTrendChart._downsample(long, 48)
    assert len(out) == 48                                          # bounded
    assert min(out) >= 0 and max(out) <= 99                        # bucket means stay in range


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


def test_yield_panel_unavailable_state(tk_root):
    """stats=None (the loader itself raised) must stay blank -- never a
    fabricated 0% or 0-count that would read like a genuinely empty window
    (CLAUDE.md: "a failure must never look like a result"). The page's own
    banner names the failure; this panel just stays at its built-in '—'."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.yield_panel import YieldPanel
    p = YieldPanel(tk_root, theme=ThemeManager(), title="Trim analysis yield")
    p.set_yield({"passed": 4, "warnings": 0, "failed": 0, "errors": 0, "untrimmed": 0,
                 "gradeable": 4, "total": 4, "pass_rate": 100.0, "trend": []},
                total_label="4 trim records")
    p.set_unit_yield({"gradeable_units": 4, "first_pass_yield": 100.0,
                      "final_yield": 100.0, "attempts_per_section": 1.0, "rework_units": 0})
    p.set_yield(None, total_label="")             # the next load fails outright
    assert p._rate.cget("text") == "—"
    assert p._counts.cget("text") == ""
    assert p._total.cget("text") == ""
    assert p._unit_line.cget("text") == ""         # the STALE prior unit-yield line is gone too
    assert "0" not in p._rate.cget("text")


def test_worst_models_list_rows_and_click(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
    got = []
    w = WorstModelsList(tk_root, theme=ThemeManager(), on_row_click=got.append)
    w.set_rows([{"model": "BAD", "units": 5, "trim_rate": 60.0, "ft_rate": 48.0},
                {"model": "OK", "units": 9, "trim_rate": 95.0, "ft_rate": None}], total=2)
    assert len(w._row_widgets) == 2
    assert [r["model"] for r in w._rows] == ["BAD", "OK"]
    w._row_widgets[0]._on_click_all()              # blocks.row's own test hook
    assert got == ["BAD"]


def test_worst_models_list_row_draws_model_statement_readout(tk_root):
    """blocks.row shape (design doc §4): model mono, a statement, a readout
    mono on the right -- the readout is Trim % (what this list ranks by), and
    a big gap earns a plain-word tag rather than a colour alone."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.worst_models_list import WorstModelsList
    w = WorstModelsList(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    w.set_rows([{"model": "OVERKILL", "units": 10, "trim_rate": 40.0, "ft_rate": 90.0}], total=1)
    texts = _labels_text(w._row_widgets[0])
    assert "OVERKILL" in texts                     # model
    assert "40%" in texts                           # readout: the trim_rate this list ranks by
    assert "overkill" in texts                      # gap = 40 - 90 = -50, tagged (not just coloured)


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


def test_dashboard_is_still_reachable(make_app):
    """Home took the landing slot on 2026-08-31 (app-shape spec §1). Dashboard
    was DE-EMPHASIZED, not deleted — its retirement is an explicit later call,
    so it stays registered and one show_page away."""
    app = make_app()
    assert app.page_container.current_page == "home"
    assert app.page_container.get_page("dashboard") is not None
    app.show_page("dashboard")
    assert app.page_container.current_page == "dashboard"


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
    assert any(r["model"] == "DASH" for r in page._worst._rows)


def test_dashboard_row_click_routes_to_model(make_app):
    app = make_app()
    page = app.page_container.get_page("dashboard")
    page._on_model_click("ROUTED")
    assert app.page_container.current_page == "model"
    # The real Model page consumes the route on show and lands on the model.
    assert app.page_container.get_page("model")._current_model == "ROUTED"
    assert app._model_route is None


def test_unit_yield_first_pass_final_and_sections(tmp_path):
    """QA audit 2026-07-13: unit basis must separate first-pass from final
    yield, treat 1P/1R + _TA_/_TB_ as parallel SECTIONS (not retrims), count
    true retrims within a section, and order same-day attempts by the
    filename timestamp."""
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.core.yield_stats import compute_unit_yield

    db = DatabaseManager(tmp_path / "u.db")
    now = datetime(2026, 5, 5)

    def add(uid, serial, fname, status, i):
        with db.session() as s:
            s.add(DBAR(filename=fname, file_path=f"/f/{i}", file_hash=f"h{i}".ljust(64, "0"),
                       model="M", serial=serial, system=SystemType.A, file_date=now,
                       timestamp=now, overall_status=status, has_multi_tracks=False,
                       processing_time=0.1, unit_id=uid))
            s.commit()

    # Unit A: dual-section 1P/1R, both pass first time -> first-pass unit.
    add("M/1/2026-05-05", "1P", "M_1P_TEST DATA_5-5-2026_8-00 AM.xls", StatusType.PASS, 1)
    add("M/1/2026-05-05", "1R", "M_1R_TEST DATA_5-5-2026_8-10 AM.xls", StatusType.WARNING, 2)
    # Unit B: TA fails at 9:00, retrimmed to PASS at 9:30 (file order reversed
    # on purpose: the timestamp in the NAME must win) -> final-only unit.
    add("M/2/2026-05-05", "2", "M_2_TA_Test Data_5-5-2026_9-30 AM.xls", StatusType.PASS, 3)
    add("M/2/2026-05-05", "2", "M_2_TA_Test Data_5-5-2026_9-00 AM.xls", StatusType.FAIL, 4)
    add("M/2/2026-05-05", "2", "M_2_TB_Test Data_5-5-2026_9-05 AM.xls", StatusType.PASS, 5)
    # Unit C: single section, fails and stays failed.
    add("M/3/2026-05-05", "3", "M_3_TEST DATA_5-5-2026_10-00 AM.xls", StatusType.FAIL, 6)

    u = compute_unit_yield(db, None, model="M")
    assert u["gradeable_units"] == 3
    assert u["first_pass_yield"] == pytest.approx(100 / 3)   # only unit A
    assert u["final_yield"] == pytest.approx(200 / 3)        # A and B
    assert u["rework_units"] == 1                            # unit B's TA
    # 5 sections, 6 attempts -> 1.2 trims/section
    assert u["attempts_per_section"] == pytest.approx(6 / 5)

    # Cohort: cutoff after the day excludes all three units.
    u2 = compute_unit_yield(db, datetime(2026, 6, 1), model="M")
    assert u2["gradeable_units"] == 0 and u2["first_pass_yield"] is None


# ---- Task 6: every failed loader is named, never drawn as zero -----------
#
# Before this task, _query wrapped compute_yield(trim) + compute_yield(ft) +
# worst_models_by_yield in ONE try/except with no logger call at all -- any of
# the three raising rendered an all-zero dict that looked like a genuinely
# empty (but healthy) window. Each of the three loaders below now fails on
# its own, is logged, and is named in ONE banner (`page._load_banner`, same
# shape as ModelPage's own -- see tests/test_model_page_failures.py).

def test_dashboard_yield_failure_names_banner_and_blanks_only_yield(make_app, monkeypatch, caplog):
    """compute_yield raising must be NAMED (not silently zeroed), must leave
    both yield panels at their blank '—' (never a fabricated rate or count),
    and must NOT blank company trend / priorities, which have real data of
    their own here."""
    import laser_trim_analyzer.gui.v6.pages.dashboard_page as dp
    app = make_app()
    now = datetime.now()
    with app.db.session() as s:
        for _ in range(5):
            _add_ar(s, "DASH", StatusType.PASS, now)
        _add_ft(s, "DASH", StatusType.FAIL, now)      # real data for "priorities"
        s.commit()

    def _boom(*a, **kw):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(dp, "compute_yield", _boom)

    page = app.page_container.get_page("dashboard")
    with caplog.at_level(logging.ERROR):
        page.reload_now()
    assert "Dashboard: yield query failed" in caplog.text

    assert page._load_banner.winfo_manager() != ""
    banner_text = page._load_banner.cget("text")
    assert "yield" in banner_text
    assert "company trend" not in banner_text
    assert "priorities" not in banner_text

    # No zero anywhere in either panel -- blank, not fabricated.
    assert page._trim_panel._rate.cget("text") == "—"
    assert page._trim_panel._counts.cget("text") == ""
    assert page._trim_panel._total.cget("text") == ""
    assert page._ft_panel._rate.cget("text") == "—"
    assert page._ft_panel._counts.cget("text") == ""

    assert page._caption.cget("text") == "Laser — · final test — over the last 90 days"
    # Priorities is untouched by the yield failure -- real FT-fail data, not
    # the "No final-test failures" empty state.
    assert "No final-test failures" not in page._priorities._cap.cget("text")


def test_dashboard_company_trend_failure_names_banner_others_unaffected(make_app, monkeypatch, caplog):
    app = make_app()
    now = datetime.now()
    with app.db.session() as s:
        for _ in range(3):
            _add_ar(s, "DASH", StatusType.PASS, now)
        _add_ar(s, "DASH", StatusType.FAIL, now)
        s.commit()

    def _boom(*a, **kw):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(app.db, "get_company_yield_trend", _boom)

    page = app.page_container.get_page("dashboard")
    with caplog.at_level(logging.ERROR):
        page.reload_now()
    assert "Dashboard: company trend failed" in caplog.text

    banner_text = page._load_banner.cget("text")
    assert "company trend" in banner_text
    assert "yield" not in banner_text
    assert "priorities" not in banner_text

    # The yield loader ran fine: a real rate, not blanked by this failure.
    assert page._trim_panel._rate.cget("text") != "—"
    assert page._caption.cget("text") == "Laser 75% · final test — over the last 90 days"


def test_dashboard_priorities_failure_names_banner_others_unaffected(make_app, monkeypatch, caplog):
    import laser_trim_analyzer.gui.v6.pages.dashboard_page as dp
    app = make_app()
    now = datetime.now()
    with app.db.session() as s:
        for _ in range(3):
            _add_ar(s, "DASH", StatusType.PASS, now)
        _add_ar(s, "DASH", StatusType.FAIL, now)
        s.commit()

    def _boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(dp, "compute_cost_priorities", _boom)

    page = app.page_container.get_page("dashboard")
    with caplog.at_level(logging.ERROR):
        page.reload_now()
    assert "Dashboard: cost priorities failed" in caplog.text

    banner_text = page._load_banner.cget("text")
    assert "priorities" in banner_text
    assert "company trend" not in banner_text
    assert "yield" not in banner_text

    assert page._trim_panel._rate.cget("text") != "—"
    assert page._caption.cget("text") == "Laser 75% · final test — over the last 90 days"


def test_dashboard_healthy_reload_has_no_load_banner(make_app):
    app = make_app()
    now = datetime.now()
    with app.db.session() as s:
        _add_ar(s, "DASH", StatusType.PASS, now)
        s.commit()
    page = app.page_container.get_page("dashboard")
    page.reload_now()
    assert page._load_banner.winfo_manager() == ""


def test_dashboard_caption_reads_both_yields_in_words(make_app):
    """Design doc §4's own example: 'Laser 61% · final test 83% over the last
    90 days'. Seeded here to 75%/50% exactly, so the assertion pins the whole
    sentence, not just a substring."""
    app = make_app()
    now = datetime.now()
    with app.db.session() as s:
        for _ in range(3):
            _add_ar(s, "DASH", StatusType.PASS, now)
        _add_ar(s, "DASH", StatusType.FAIL, now)          # trim: 3/4 = 75%
        _add_ft(s, "DASH", StatusType.PASS, now)
        _add_ft(s, "DASH", StatusType.FAIL, now)          # FT: 1/2 = 50%
        s.commit()
    page = app.page_container.get_page("dashboard")
    page.reload_now()
    assert page._caption.cget("text") == "Laser 75% · final test 50% over the last 90 days"


def test_dashboard_caption_dash_when_no_data_yet(make_app):
    """An empty database is not a failure (no banner) -- but a yield that is
    genuinely unknown still reads '—' in the caption, same as a failed load
    (brief: "and '—' when a yield is unknown")."""
    app = make_app()
    page = app.page_container.get_page("dashboard")
    page.reload_now()
    assert page._caption.cget("text") == "Laser — · final test — over the last 90 days"
    assert page._load_banner.winfo_manager() == ""
