"""Spec 3b — Triage. Foundations §4.1/§4.3. Fixtures in tests/conftest.py."""
from datetime import datetime, timedelta

import pytest

# ---- Task 1: helpers ------------------------------------------------------

def test_metric_label_humanizes():
    from laser_trim_analyzer.ml.drift_types import metric_label
    assert metric_label("untrimmed_resistance") == "Untrimmed resistance"
    assert metric_label("linearity_error") == "Linearity error"
    assert metric_label("measured_electrical_angle") == "Electrical angle"
    assert metric_label("totally_unknown") == "totally_unknown"  # graceful passthrough


def test_list_known_models_empty(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import list_known_models
    assert list_known_models(DatabaseManager(tmp_path / "e.db")) == []


def _add_ar(s, model, when):
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, SystemType, StatusType)
    s.add(DBAR(filename=f"{model}.xls", file_path=f"/f/{model}.xls", file_hash=f"h{model}{when.microsecond}",
               model=model, serial="sn1", system=SystemType.A, file_date=when, timestamp=when,
               overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))


def test_list_known_models_distinct(tmp_path):
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.manager import list_known_models
    db = DatabaseManager(tmp_path / "d.db")
    with db.session() as s:
        for m in ("8340-1", "8232-1", "8877"):
            _add_ar(s, m, datetime.now())
        s.commit()
    assert {x.model for x in list_known_models(db)} == {"8340-1", "8232-1", "8877"}


def test_list_known_models_includes_smoothness_only(tmp_path):
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.database.models import SmoothnessResult as DBSR, StatusType
    from laser_trim_analyzer.ml.manager import list_known_models
    db = DatabaseManager(tmp_path / "s.db")
    with db.session() as s:
        s.add(DBSR(filename="s.xls", file_path="/f/s.xls", file_hash="hs", file_date=datetime.now(),
                   model="SMOOTH-ONLY", serial="sn1", test_date=datetime.now(),
                   overall_status=StatusType.PASS, timestamp=datetime.now()))
        s.commit()
    assert "SMOOTH-ONLY" in {x.model for x in list_known_models(db)}


def test_list_known_models_tier_merged_from_drift_api(tmp_path, monkeypatch):
    """Tier comes from a SINGLE get_drifting_models call; others default STABLE.
    Mock the drift API so the test is deterministic (not coupled to detector math)."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.drift_types import (
        AlertType, DriftTier, ModelAlertSummary)
    import laser_trim_analyzer.ml.manager as mgr
    db = DatabaseManager(tmp_path / "t.db")
    with db.session() as s:
        _add_ar(s, "FLAGGED", datetime.now())
        _add_ar(s, "CALM", datetime.now())
        s.commit()
    monkeypatch.setattr(mgr, "get_drifting_models", lambda _db, *a, **k: [
        ModelAlertSummary(model="FLAGGED", tier=DriftTier.DRIFT,
                          alert_type=AlertType.STEP_CHANGE,
                          worst_metric="untrimmed_resistance", magnitude=4.2)])
    by = {x.model: x.tier for x in mgr.list_known_models(db)}
    assert by["FLAGGED"] == DriftTier.DRIFT
    assert by["CALM"] == DriftTier.STABLE


def test_list_known_models_single_query_no_per_model_status(tmp_path, monkeypatch):
    """Regression guard for the N+1 bug: list_known_models must NOT call
    get_model_drift_status once per model."""
    from datetime import datetime
    from laser_trim_analyzer.database.manager import DatabaseManager
    import laser_trim_analyzer.ml.manager as mgr
    db = DatabaseManager(tmp_path / "n.db")
    with db.session() as s:
        for i in range(5):
            _add_ar(s, f"M{i}", datetime.now())
        s.commit()
    calls = {"n": 0}
    real = mgr.get_model_drift_status
    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(mgr, "get_model_drift_status", counted)
    mgr.list_known_models(db)
    assert calls["n"] == 0  # tiers come from get_drifting_models, not per-model status


# ---- Task 2: triage alert ordering — REMOVED 2026-08-31 -------------------
# Two tests lived here for the σ-alert ordering helper. The comment above them
# claimed the feed "still feeds the drift table and the v5 pages"; a grep for
# call sites across src/, scripts/ and tests/ found none — the FOCUS list had
# taken the only surface on 2026-08-29 and nothing replaced the caller. The
# feed, its ordering helper and these two tests were deleted together. What the
# tests protected (worst tier on top) still holds where it is still used: the
# detector's own worst-metric pick in multi_metric_drift_detector.py, and the
# FOCUS ordering asserted in test_focus_list_zone.py and the sweep.


# ---- Task 4: BrowseZone ---------------------------------------------------

def _ms(model, tier=None):
    from laser_trim_analyzer.ml.drift_types import DriftTier, ModelSummary
    return ModelSummary(model=model, tier=tier or DriftTier.STABLE)


def test_the_browse_legend_makes_no_claim_about_order(tk_root):
    """I5 (final review, 2026-09-24): the legend said "Status = drift tier, worst first" over an
    ALPHABETICAL list. Ruling: the browse list is the lookup list ("Needs a look" is the ranked
    one), so the order claim goes, the rest of the key stays."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    z.set_models([_ms("B2"), _ms("A1")])
    legend = z._legend.cget("text")
    assert "worst first" not in legend
    # F5 review: "Date = last processed" beside "Inactive · last trimmed" read as a contradiction.
    assert legend.startswith("Status = drift tier. Date = the model's newest laser or smoothness "
                             "file of any kind")
    assert [r._summary.model for r in z._rows] == ["B2", "A1"]   # as given -- no ranking here


def test_browse_one_row_per_model(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    z.set_models([_ms(f"M{i}") for i in range(5)])
    assert len(z._rows) == 5


def test_browse_filter_substring(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    # Substring filter: "83" must appear as consecutive chars. 8340-1 and 8830-1
    # contain "83"; 8877 does not. (Plan's original datum "8232-1" was a typo —
    # 8-2-3-2 has no consecutive "83" — inconsistent with the substring impl.)
    z.set_models([_ms("8340-1"), _ms("8830-1"), _ms("8877")])
    z.set_filter("83")
    # _summary: a test hook browse_zone.py stashes on the blocks.row frame it builds, the same
    # way blocks.row's own _on_click_all/_set_hover/_on_leave are stashed for tests.
    shown = {r._summary.model for r in z._rows}
    assert shown == {"8340-1", "8830-1"}


def test_browse_row_click_emits_model(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone
    got = []
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=got.append)
    z.set_models([_ms("CLICKED")])
    z._rows[0]._on_click_all()          # blocks.row's own click test hook
    assert got == ["CLICKED"]


def test_browse_discloses_cap(tk_root):
    """Q10: when more than the render cap exist, say so instead of silently truncating."""
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.browse_zone import BrowseZone, ROW_CAP
    z = BrowseZone(tk_root, theme=ThemeManager(), on_row_click=lambda _: None)
    z.set_models([_ms(f"M{i:04d}") for i in range(ROW_CAP + 25)])
    assert len(z._rows) == ROW_CAP
    assert "Showing" in z._cap_label.cget("text") and str(ROW_CAP + 25) in z._cap_label.cget("text")


# ---- Task 5: TriagePage + routing ----------------------------------------
# Fixtures build the FOCUS list through the REAL DatabaseManager and the real
# compute path (no monkeypatched summaries): the point of the 2026-08-29
# rewire is that the page renders exactly what `compute_focus_list` decided,
# so a mocked feed would test nothing that ships.

D0 = datetime(2026, 1, 5)      # SPC anchors on the DATA's newest date, not "now"


def _labels(widget):
    """Every CTkLabel text under `widget` — what the page actually SAYS."""
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels(c))
    return out


def _add_lot(db, model, day, n, fails):
    """One production lot: n gradeable units on one file_date, `fails` of them FAIL."""
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, StatusType, SystemType)
    with db.session() as s:
        for i in range(n):
            # `system` is NOT NULL in the schema; the SPC query never reads it.
            s.add(DBAR(model=model, serial=f"{model}-{day:%m%d}-{i}", system=SystemType.A,
                       filename=f"{model}_{i}_{day:%m-%d-%Y}.xls", file_date=day,
                       overall_status=StatusType.FAIL if i < fails else StatusType.PASS))


def _seed(db, model, n_lots=12, fails_last=0, start=D0, n_per=20, base_fails=2):
    """Weekly lots at a steady baseline rate, then one lot at `fails_last`."""
    for k in range(n_lots - 1):
        _add_lot(db, model, start + timedelta(days=7 * k), n_per, base_fails)
    last_day = start + timedelta(days=7 * (n_lots - 1))
    _add_lot(db, model, last_day, n_per, fails_last if fails_last else base_fails)
    return last_day


def _drifting_app(make_app):
    """App whose DB holds one drifting CURRENT model and one long-dormant one.

    HOT owns the anchor and blew out on its last lot (10% -> 60%); OLD stopped
    running ~300 days earlier, so it is neither on the FOCUS list (compute's
    own ACTIVE_DAYS rule) nor in the Active browse scope.
    """
    app = make_app()
    _seed(app.db, "HOT", fails_last=12)
    _seed(app.db, "OLD", start=D0 - timedelta(days=300))
    triage = app.page_container.get_page("triage")
    triage.reload_now()       # synchronous path for tests
    return app, triage


def test_v6app_consume_model_route(make_app):
    app = make_app()
    assert app.consume_model_route() is None
    app.set_model_route("M", "linearity_error")
    assert app.consume_model_route() == "M"       # 3b consumes model only
    assert app.consume_model_route() is None       # one-shot


def test_consume_model_tab_is_popped_separately_from_the_model_route(make_app):
    """Task 7: a route can also name a tab to land on (e.g. the Findings page's
    opened-row button routes to the model's Findings tab). It is stored apart from
    (model, focus_metric) so consuming one never consumes the other, and it is a
    one-shot like every other routing hint."""
    app = make_app()
    assert app.consume_model_tab() is None
    app.set_model_route("M", tab="findings")
    assert app.consume_model_tab() == "findings"
    assert app.consume_model_tab() is None                 # one-shot
    assert app.consume_model_route() == "M"                # the model route is untouched


def test_set_model_route_without_a_tab_clears_any_earlier_tab_route(make_app):
    """A stale tab from a PREVIOUS navigation must never leak into a later one that
    didn't ask for it -- e.g. Triage's plain set_model_route(model, focus) after a
    Findings-page visit must not still land on the Findings tab."""
    app = make_app()
    app.set_model_route("M", tab="findings")
    app.set_model_route("N")
    assert app.consume_model_tab() is None


def test_triage_focus_zone_shows_the_drifting_model(make_app):
    app, triage = _drifting_app(make_app)
    # The zone's own heading is suppressed (show_heading=False) -- Triage draws its own
    # "Needs a look" blocks.group_header instead, in _apply(); its count pill is a SEPARATE
    # label from the title (count_pill(), not text baked into one string), so check both are
    # present rather than one substring.
    header_texts = _labels(triage._focus_header)
    assert "Needs a look" in header_texts and "1" in header_texts
    assert [r.entry.model for r in triage._focus._rows] == ["HOT"]


def test_triage_focus_click_routes_to_model_with_spc_metric(make_app):
    app, triage = _drifting_app(make_app)
    triage._focus._rows[0]._on_click()
    assert app.page_container.current_page == "model"
    # The real Model page consumes the route on show — the deep-link lands on
    # HOT with the metric that put it on the list preselected.
    model_page = app.page_container.get_page("model")
    assert model_page._current_model == "HOT"
    assert model_page._current_metric == "linearity_fail_fraction"
    assert app._model_route is None


def test_scope_toggle_filters_browse_only(make_app):
    """FOCUS membership belongs to compute_focus_list; the toggle is a browse filter."""
    app, triage = _drifting_app(make_app)
    n_focus = len(triage._focus._rows)
    assert {r._summary.model for r in triage._browse._rows} == {"HOT"}   # Active scope
    triage._on_scope_change("All models")
    assert {r._summary.model for r in triage._browse._rows} == {"HOT", "OLD"}
    assert len(triage._focus._rows) == n_focus == 1
    assert [r.entry.model for r in triage._focus._rows] == ["HOT"]
    triage._on_scope_change("Active")
    assert {r._summary.model for r in triage._browse._rows} == {"HOT"}
    assert len(triage._focus._rows) == n_focus


def test_triage_reload_now_populates(make_app):
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR, SystemType, StatusType
    app = make_app()
    with app.db.session() as s:
        s.add(DBAR(filename="x.xls", file_path="/f/x.xls", file_hash="hx", model="LOAD-TEST",
                   serial="sn1", system=SystemType.A, file_date=datetime.now(), timestamp=datetime.now(),
                   overall_status=StatusType.PASS, has_multi_tracks=False, processing_time=0.1))
        s.commit()
    triage = app.page_container.get_page("triage")
    triage.reload_now()       # synchronous path for tests
    assert "LOAD-TEST" in _labels(triage)


def test_triage_empty_db_shows_within_tolerance(make_app):
    """No data at all must read as 'nothing to look at', never as a blank zone."""
    app = make_app()
    triage = app.page_container.get_page("triage")
    triage.reload_now()
    header_texts = _labels(triage._focus_header)
    assert "Needs a look" in header_texts and "0" in header_texts
    assert any("within tolerance" in t for t in _labels(triage._focus))


def test_the_browse_list_is_not_squeezed_out_at_1280_by_720(make_app):
    """The one known clip carried in from facelift step 1's review: the FOCUS list's fixed
    height (320px, regardless of window size) left the browse list squeezed to nothing on a
    short window -- 48 of its text widgets unmapped entirely. Seeds enough drifting models to
    fill FOCUS_CAP (a full-height focus zone) plus enough plain ones for a real browse list,
    then checks with the audit's own detector (render_pages.py) on a real mapped window
    (invisible: alpha 0), the same technique test_spec3f_home.py's own 1280x720 test uses."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from render_pages import find_clipped_text_widgets

    app = make_app()
    for i in range(9):                                    # > FOCUS_CAP (7): a full focus zone
        _seed(app.db, f"DRIFT-{i}", fails_last=12)
    for i in range(15):                                    # stay in control -> browse-only rows
        _seed(app.db, f"PLAIN-{i}", fails_last=2, base_fails=2)
    triage = app.page_container.get_page("triage")
    app.show_page("triage")
    triage.reload_now()
    try:
        app.attributes("-alpha", 0.0)
    except Exception:
        pass
    app.geometry("1280x720+20000+20000")
    app.deiconify()
    app.update_idletasks()
    app.update()
    try:
        hits = find_clipped_text_widgets(triage, page="triage", window_size="1280x720")
        assert not hits, [h.line() for h in hits]
    finally:
        app.withdraw()


@pytest.mark.parametrize("size", ((1280, 720), (960, 640)))
@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_the_browse_list_keeps_its_minimum_at_every_scaling(make_app, scale, size):
    """Final review (2026-09-24): _fit_focus_zone mixed its units -- winfo_height() is REAL
    pixels, _BROWSE_MIN_H and the theme's spacing are CustomTkinter's unscaled units, and
    configure(height=) scales its argument again. At 150% (the Windows laptop's likely setting)
    the focus zone took its full 320 (480 real px) and the browse list's guaranteed 200 (300
    real px) shrank to a sliver.

    Windows applies the DPI factor to BOTH scalings, so a 1280x720 window there is 1920x1080
    real pixels at 150%; the window is borderless (overrideredirect) so macOS does not clamp a
    window that big back to this Mac's screen. Measured in real pixels on a mapped window, at
    the audited 1280x720 and at 960x640 -- the app's own minimum size, where the page is short
    enough for the old arithmetic to starve the list (at 1280x720 it happened to have room)."""
    import customtkinter as ctk
    from laser_trim_analyzer.gui.v6.pages import triage_page as tp

    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    try:
        app = make_app()
        for i in range(9):                                  # > FOCUS_CAP (7): a full focus zone
            _seed(app.db, f"DRIFT-{i}", fails_last=12)
        for i in range(15):
            _seed(app.db, f"PLAIN-{i}", fails_last=2, base_fails=2)
        triage = app.page_container.get_page("triage")
        app.show_page("triage")
        triage.reload_now()
        try:
            app.attributes("-alpha", 0.0)
        except Exception:
            pass
        app.overrideredirect(True)
        width, height = size
        app.geometry(f"{width}x{height}+20000+20000")
        app.deiconify()
        for _ in range(3):
            app.update_idletasks()
            app.update()
        try:
            assert app.winfo_height() == round(height * scale)    # really `height` units tall
            browse_real = triage._browse.winfo_height()
            assert browse_real >= tp._BROWSE_MIN_H * scale - 2, (
                f"browse list {browse_real} px at {scale:.0%}: its guaranteed minimum is "
                f"{tp._BROWSE_MIN_H} units = {tp._BROWSE_MIN_H * scale:.0f} px")
            # and the focus zone is asked for a height in the unit configure() expects
            assert tp._FOCUS_ZONE_MIN_H <= triage._focus_wrap.cget("height") <= tp._FOCUS_ZONE_MAX_H
        finally:
            app.withdraw()
    finally:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)



# ---- final review (2026-09-24, I2): a failed load is named, never drawn as a zero ---------------

def _count_pills(header):
    return [t for t in _labels(header) if str(t).replace(",", "").isdigit()]


def test_a_focus_crash_is_a_banner_with_no_count_pill(make_app, monkeypatch):
    import laser_trim_analyzer.gui.v6.focus_data as fd

    def boom(_db):
        raise RuntimeError("invented focus crash")
    monkeypatch.setattr(fd, "compute_focus_list", boom)
    app, triage = _drifting_app(make_app)
    assert triage._load_banner.winfo_manager() == "pack"
    text = triage._load_banner.cget("text")
    assert "drifting" in text and "RuntimeError: invented focus crash" in text
    assert "Needs a look" in _labels(triage._focus_header)
    assert _count_pills(triage._focus_header) == []          # never "Needs a look · 0"
    zone = " ".join(_labels(triage._focus))
    assert "within tolerance" not in zone and "Unavailable" in zone
    assert {r._summary.model for r in triage._browse._rows} == {"HOT"}   # browse unaffected


def test_a_failed_model_list_is_a_banner_with_no_count_pill(make_app, monkeypatch):
    import laser_trim_analyzer.gui.v6.pages.triage_page as tp

    def boom(*a, **k):
        raise RuntimeError("invented inventory crash")
    monkeypatch.setattr(tp, "list_known_models", boom)
    app, triage = _drifting_app(make_app)
    text = triage._load_banner.cget("text")
    assert triage._load_banner.winfo_manager() == "pack"
    assert "model list" in text and "RuntimeError: invented inventory crash" in text
    assert "All models" in _labels(triage._browse._header)
    assert _count_pills(triage._browse._header) == []        # never "All models · 0"
    assert "Unavailable" in " ".join(_labels(triage._browse))
    triage._on_scope_change("All models")                      # the toggle keeps it unavailable
    assert _count_pills(triage._browse._header) == []


def test_a_good_triage_load_after_a_failure_clears_the_banner(make_app, monkeypatch):
    import laser_trim_analyzer.gui.v6.pages.triage_page as tp

    def boom(*a, **k):
        raise RuntimeError("invented inventory crash")
    monkeypatch.setattr(tp, "list_known_models", boom)
    app, triage = _drifting_app(make_app)
    assert triage._load_banner.winfo_manager() == "pack"
    monkeypatch.undo()
    triage.reload_now()
    assert triage._load_banner.winfo_manager() == ""
    assert _count_pills(triage._focus_header) == ["1"]
    assert _count_pills(triage._browse._header) == ["1"]


def test_the_failure_banner_never_squeezes_the_browse_list(make_app, monkeypatch):
    """The banner takes room above the focus zone; the fit has to count it, or at the app's
    minimum 960x640 the browse list drops below its guaranteed minimum."""
    import laser_trim_analyzer.gui.v6.focus_data as fd
    from laser_trim_analyzer.gui.v6.pages import triage_page as tp

    def boom(_db):
        raise RuntimeError("invented focus crash " + "with a long explanation " * 6)
    monkeypatch.setattr(fd, "compute_focus_list", boom)
    app = make_app()
    for i in range(15):
        _seed(app.db, f"PLAIN-{i}", fails_last=2, base_fails=2)
    triage = app.page_container.get_page("triage")
    app.show_page("triage")
    triage.reload_now()
    try:
        app.attributes("-alpha", 0.0)
    except Exception:
        pass
    app.overrideredirect(True)
    app.geometry("960x640+20000+20000")
    app.deiconify()
    for _ in range(3):
        app.update_idletasks()
        app.update()
    try:
        assert triage._load_banner.winfo_manager() == "pack"
        assert triage._browse.winfo_height() >= tp._BROWSE_MIN_H - 2, triage._browse.winfo_height()
    finally:
        app.withdraw()


# ---- facelift F4 (2026-09-25): an older load never overwrites a newer one -----------------------

@pytest.mark.parametrize("newer", ("async", "sync"))
def test_an_older_triage_load_never_overwrites_a_newer_one(make_app, monkeypatch, newer):
    """Triage's on_show load runs on a thread and applies through safe_after, in whatever order the
    loads FINISH. An older, slower load landing after a newer one put its state back over the newer
    one -- here a healthy list over a crash the newer load had named. Dropped now unless newest."""
    from test_spec3f_home import _older_then_newer, _pump_ui, _pump_until, _settle_workers
    import laser_trim_analyzer.gui.v6.pages.triage_page as tp
    from laser_trim_analyzer.gui.v6.focus_data import FocusLoadFailed
    from laser_trim_analyzer.ml.spc import FocusResult
    app = make_app()
    _seed(app.db, "PLAIN-0", fails_last=2, base_fails=2)
    triage = app.page_container.get_page("triage")
    _settle_workers(app)
    load = _older_then_newer(
        lambda: (FocusResult(focus=[], chronic=[], anchor=None), D0),               # older: healthy
        lambda: (FocusLoadFailed(focus=[], chronic=[], anchor=None,
                                 error="RuntimeError: invented focus crash"), D0))  # newer: crashed
    monkeypatch.setattr(tp, "load_focus", load)
    triage.on_show()                           # the older load, still in its query...
    load.started()
    if newer == "async":
        triage.on_show()                       # ...when a newer one starts and finishes first
        assert _pump_until(app, lambda: triage._load_banner.winfo_manager() == "pack")
    else:
        triage.reload_now()
    assert triage._load_banner.winfo_manager() == "pack"
    load.release()                             # the older load finishes LAST
    _pump_ui(app)
    assert triage._load_banner.winfo_manager() == "pack", "an older load overwrote a newer one"
    assert "invented focus crash" in triage._load_banner.cget("text")
    assert _count_pills(triage._focus_header) == []          # still no count over a crash


# ---- facelift F4 (2026-09-25): the failure banner wraps to the page ---------------------------

@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_the_failure_banner_wraps_to_the_page(make_app, monkeypatch, scale):
    """Re-review Minor 2: Triage's banner was req=1015 alloc=756 px at 960x640 (100%) and
    req=1519 alloc=1134 at 150% -- cut. The audit's own detector: nothing on the page cut at
    1280x720, and the banner not cut at 960x640 (the rest of that size is TRACKER D7's)."""
    import customtkinter as ctk
    import laser_trim_analyzer.gui.v6.focus_data as fd
    import laser_trim_analyzer.gui.v6.pages.triage_page as tp
    from test_blocks import failure_texts_cut

    long = ("invented crash with a long explanation that goes on for a while " * 2).strip()

    def boom(*_a, **_k):
        raise RuntimeError(long)
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    try:
        monkeypatch.setattr(fd, "compute_focus_list", boom)
        monkeypatch.setattr(tp, "list_known_models", boom)
        app = make_app()
        triage = app.page_container.get_page("triage")
        app.show_page("triage")
        triage.reload_now()
        assert triage._load_banner.winfo_manager() == "pack"
        cut = failure_texts_cut(app, triage, [triage._load_banner])
        assert cut["1280x720"] == ([], []), cut["1280x720"]
        assert not cut["960x640"][0], cut["960x640"][0]
    finally:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)


# ---- F5 (2026-09-25): "Inactive" models -- labelled, never hidden ------------------------------

def _statement(row):
    """blocks.row's statement: the first label in its middle column."""
    import customtkinter as ctk
    mid = row.winfo_children()[1]
    return next(w.cget("text") for w in mid.winfo_children() if isinstance(w, ctk.CTkLabel))


def test_all_models_reads_inactive_with_its_month_where_a_drift_tier_would_mean_nothing(make_app):
    from test_findings_page import _seed_inactive
    app = make_app()
    tag = _seed_inactive(app)
    page = app.page_container.get_page("triage")
    page._on_scope_change("All models")
    page.reload_now()
    status = {r._summary.model: _statement(r) for r in page._browse._rows}
    assert status["OLD"] == tag
    assert status["LIVE"] == "Stable"
    assert "Inactive" in page._browse._legend.cget("text")


def test_triage_says_so_when_which_models_are_inactive_cannot_be_read(make_app, monkeypatch):
    import laser_trim_analyzer.gui.v6.pages.triage_page as tp
    from test_findings_page import _seed_inactive

    def boom(db):
        raise RuntimeError("invented activity crash")
    monkeypatch.setattr(tp, "load_activity", boom)
    app = make_app()
    _seed_inactive(app)
    page = app.page_container.get_page("triage")
    page._on_scope_change("All models")
    page.reload_now()
    assert page._load_banner.winfo_manager() == "pack"
    assert "Which models are inactive could not be worked out" in page._load_banner.cget("text")
    assert {r._summary.model: _statement(r) for r in page._browse._rows}["OLD"] == "Stable"


# ---- F5 review (Important 2): the date column says what it is -------------------------------------
# It is the model's newest laser or smoothness file of ANY kind (ml/manager.list_known_models), so an
# inactive model's date can be later than its last trim: 8275's trim files end 2024-08-27 while its
# smoothness tests run to 2026-08-26 (and its final tests to 2026-09-21, which the column never
# counts). "Inactive · last trimmed Aug 2024" beside "2026-08-26" must read as two facts.

def test_an_inactive_rows_date_is_named_the_newest_file_of_any_kind(make_app):
    from datetime import datetime
    from laser_trim_analyzer.database.models import (
        FinalTestResult, SmoothnessResult, StatusType)
    from test_model_activity import NEWEST, _file
    app = make_app()
    _file(app.db, "LIVE", NEWEST)
    _file(app.db, "OLD", datetime(2024, 8, 27, 8, 31))
    with app.db.session() as s:
        s.add(SmoothnessResult(filename="old_smooth.xls", model="OLD", serial="OLD-7",
                               file_date=datetime(2026, 8, 26), overall_status=StatusType.PASS))
        s.add(FinalTestResult(filename="old_ft.xls", model="OLD", serial="OLD-8",
                              file_date=datetime(2026, 9, 21), overall_status=StatusType.PASS))
    page = app.page_container.get_page("triage")
    page._on_scope_change("All models")
    page.reload_now()
    row = next(r for r in page._browse._rows if r._summary.model == "OLD")
    assert _statement(row) == "Inactive · last trimmed Aug 2024"
    assert "2026-08-26" in _labels(row)                    # the smoothness test, never the final test
    assert "2026-09-21" not in _labels(row)
    assert "newest file, any kind" in _labels(page._browse._header)      # the column's own heading
    legend = page._browse._legend.cget("text")
    assert "not a final test" in legend and "later than its last trim" in legend
