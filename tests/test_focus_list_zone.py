"""FOCUS list zone — the row-text contract and the widget that renders it.

Deliberately NOT folded into `test_spc_chart_helpers.py`: that file is
documented as the pure draw-parameter mapping (no Tk, no rendering), and these
tests build real widgets under a Tk root. Kept apart, a Tk/render failure can
never be misread as a failure of the helper math.

Every fixture series here is REAL `build_fraction_series` output (and one is a
real `compute_focus_list` row off a seeded DB). A hand-mocked SpcSeries would
let the widget drift away from the numbers the chart draws, which is the exact
failure this redesign exists to end.
"""
from datetime import datetime, timedelta

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.focus_chart import spc_draw_params
from laser_trim_analyzer.gui.v6.widgets.focus_list_zone import (
    FOCUS_CAP, FocusListZone, focus_row_texts)
from laser_trim_analyzer.ml.spc import (
    RECENT_K, FocusEntry, FocusResult, build_fraction_series)

D0 = datetime(2026, 1, 5)


# ---- fixtures: real series, real-shaped entries ---------------------------

def _lot_samples(day, n, fails):
    """n units on one day, `fails` of them failing linearity."""
    return [(day, 1.0 if i < fails else 0.0) for i in range(n)]


def _hot_history(n_lots=12, n_per=20, base_fails=2, fails_last=12, start=D0):
    """Weekly lots at 10% fail, with the NEWEST lot blowing out to 60%."""
    out = []
    for k in range(n_lots - 1):
        out += _lot_samples(start + timedelta(days=7 * k), n_per, base_fails)
    last = start + timedelta(days=7 * (n_lots - 1))
    out += _lot_samples(last, n_per, fails_last)
    return out, last


def _older_excursion_history():
    """Excursion at lot 11 (older than the window) AND at lot 17 (newest)."""
    out = []
    for k in range(10):
        out += _lot_samples(D0 + timedelta(days=7 * k), 20, 2)
    day = D0 + timedelta(days=7 * 10)
    out += _lot_samples(day, 20, 12)                       # lot 11 — old news
    for k in range(1, 6):
        out += _lot_samples(day + timedelta(days=7 * k), 20, 2)
    last = day + timedelta(days=7 * 6)
    out += _lot_samples(last, 20, 12)                      # lot 17 — today's fire
    return out, last


def _entry(model="8340-1", *, days_ago=10, history=None, excess=12.0,
           units_per_week=140.0):
    """A FocusEntry around a real series; `days_ago` = anchor age of the last lot."""
    hist, last = history if history is not None else _hot_history()
    anchor = last + timedelta(days=days_ago)
    series = build_fraction_series(model, "linearity_fail_fraction", hist,
                                   anchor=anchor)
    assert series.judged, "fixture must produce a judged series"
    flagged = [p for p in series.points[-RECENT_K:] if p.ooc]
    assert flagged, "fixture must produce a flagged recent lot"
    p_recent = (sum(p.value * p.n for p in flagged)
                / sum(p.n for p in flagged))
    return FocusEntry(
        model=model, series=series, excess_per_week=excess,
        units_per_week=units_per_week, p_base=series.p_base, p_recent=p_recent,
        n_flagged_recent=len(flagged), last_lot_end=series.points[-1].end,
        verdict=f"failing ~{excess:.0f} more units/week than its own baseline",
        sub_line=(f"{len(flagged)} of last {RECENT_K} lots out of control"
                  f" · fail rate {series.p_base * 100:.0f}%"
                  f" → {p_recent * 100:.0f}% · ~{units_per_week:.0f} units/wk"))


def _anchor_of(entry, days_ago=10):
    return entry.last_lot_end + timedelta(days=days_ago)


def _result(entries=(), chronic=(), days_ago=10):
    entries = list(entries)
    anchor = _anchor_of(entries[0], days_ago) if entries else None
    return FocusResult(focus=entries, chronic=list(chronic), anchor=anchor)


def _labels(widget):
    """Every CTkLabel text under a widget (same walker the 3b tests use)."""
    import customtkinter as ctk
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_labels(c))
    return out


def _zone(tk_root, sink=None):
    cb = ((lambda m, metric: sink.append((m, metric))) if sink is not None
          else (lambda *_: None))
    return FocusListZone(tk_root, theme=ThemeManager(), on_row_click=cb)


# ---- focus_row_texts: the pure row model ----------------------------------

def test_row_texts_pass_the_entry_lines_through_untouched():
    """One computation: the row quotes the entry, it never re-derives a number."""
    e = _entry()
    x = focus_row_texts(e, _anchor_of(e))
    assert x["title"] == "8340-1"
    assert x["line1"] == e.verdict
    assert x["line2"] == e.sub_line


def test_row_texts_name_the_last_lot_and_its_age():
    e = _entry(days_ago=10)
    x = focus_row_texts(e, _anchor_of(e, 10))
    assert x["when"] == f"last lot {e.last_lot_end:%b %d} (10d ago)"
    assert "still open" not in x["when"]           # 10 days idle = closed


def test_row_texts_say_today_for_a_same_day_lot():
    e = _entry(days_ago=0)
    x = focus_row_texts(e, _anchor_of(e, 0))
    assert x["when"].startswith(f"last lot {e.last_lot_end:%b %d} (today)")


def test_row_texts_disclose_a_still_open_lot():
    """A lot that may still be receiving units is a preview, not a verdict."""
    e = _entry(days_ago=1)
    assert e.series.points[-1].is_open                       # fixture sanity
    x = focus_row_texts(e, _anchor_of(e, 1))
    assert x["when"] == f"last lot {e.last_lot_end:%b %d} (1d ago) · lot still open"


def test_row_texts_without_an_anchor_omit_the_age():
    """No anchor (empty DB edge) must not fabricate an age or crash."""
    e = _entry()
    x = focus_row_texts(e, None)
    assert x["when"] == f"last lot {e.last_lot_end:%b %d}"


# ---- the zone -------------------------------------------------------------

def test_zone_heading_counts_and_caption_names_the_anchor(tk_root):
    z = _zone(tk_root)
    res = _result([_entry("A"), _entry("B")])
    z.set_result(res)
    texts = _labels(z)
    assert "FOCUS — drifting now, biggest first (2)" in texts
    assert (f"as of last processed data {res.anchor:%b %d} · a model is here "
            f"only while its last {RECENT_K} lots include one outside its own "
            "control limits · ranked by extra failing units per week") in texts


def test_zone_empty_state_names_last_processed(tk_root):
    z = _zone(tk_root)
    z.set_result(FocusResult(focus=[], chronic=[], anchor=None),
                 last_processed=datetime(2026, 5, 30))
    txt = " | ".join(_labels(z))
    assert "All models within tolerance — last processed 2026-05-30." in txt
    assert "FOCUS — drifting now, biggest first (0)" in txt
    assert "None" not in txt                       # no anchor => no junk date


def test_zone_caps_at_seven_rows_and_expands(tk_root):
    z = _zone(tk_root)
    z.set_result(_result([_entry(f"M{i}") for i in range(FOCUS_CAP + 2)]))
    assert len(z._rows) == FOCUS_CAP
    # winfo_manager(), NOT winfo_ismapped(): under the withdrawn test root
    # nothing is ever mapped, so an ismapped assertion would pass vacuously.
    assert z._more_btn.winfo_manager() == "pack"
    assert z._more_btn.cget("text") == "+ 2 more models with smaller signals — show all"
    z._more_btn.invoke()
    assert len(z._rows) == FOCUS_CAP + 2
    assert "top 7" in z._more_btn.cget("text")     # toggles back the other way
    z._more_btn.invoke()
    assert len(z._rows) == FOCUS_CAP


def test_zone_expander_singular_for_one_hidden_model(tk_root):
    z = _zone(tk_root)
    z.set_result(_result([_entry(f"M{i}") for i in range(FOCUS_CAP + 1)]))
    assert z._more_btn.cget("text") == "+ 1 more model with smaller signals — show all"


def test_zone_hides_the_expander_when_everything_fits(tk_root):
    z = _zone(tk_root)
    z.set_result(_result([_entry("A")]))
    assert z._more_btn.winfo_manager() == ""      # not packed at all


def test_row_click_routes_model_and_metric(tk_root):
    got = []
    z = _zone(tk_root, got)
    z.set_result(_result([_entry("ROUTED")]))
    row = z._rows[0]
    row._on_click()
    assert got == [("ROUTED", "linearity_fail_fraction")]


def test_whole_row_is_clickable_not_just_the_name(tk_root):
    """Labels AND the sparkline carry the binding — the row is one target."""
    z = _zone(tk_root)
    z.set_result(_result([_entry("CLICKY")]))
    row = z._rows[0]
    assert row.canvas.get_tk_widget() in row._clickables
    assert row in row._clickables
    assert len(row._clickables) >= 5               # rank, name, when, 2 lines, spark
    for w in row._clickables:
        raw = getattr(w, "_label", None) or getattr(w, "_canvas", None) or w
        assert raw.bind("<Button-1>"), f"{w} has no click binding"


def test_rows_release_their_figures_on_refresh(tk_root):
    """7+ leaked Figures per refresh is a real leak in an all-shift process."""
    z = _zone(tk_root)
    z.set_result(_result([_entry(f"M{i}") for i in range(3)]))
    old_figs = [r.fig for r in z._rows]
    old_widgets = [r.canvas.get_tk_widget() for r in z._rows]
    z.set_result(_result([_entry("ONLY")]))
    assert len(z._rows) == 1
    assert all(not w.winfo_exists() for w in old_widgets)
    assert all(len(f.axes) == 0 for f in old_figs)  # cleared, not just hidden


def test_sparkline_marks_exactly_the_lots_the_chart_flags(tk_root):
    """List/chart parity by construction: both draw from spc_draw_params."""
    t = ThemeManager()
    z = _zone(tk_root)
    e = _entry("SPLIT", history=_older_excursion_history())
    z.set_result(_result([e]))
    p = spc_draw_params(e.series)
    assert p["flag_idx"] and p["old_idx"]          # fixture exercises both
    ax = z._rows[0].fig.axes[0]
    by_color = {}
    for line in ax.lines:
        by_color.setdefault(line.get_color(), []).append(list(line.get_xdata()))
    assert by_color[t.TIER_OOC] == [[float(i) for i in p["flag_idx"]]]
    assert by_color[t.TIER_WARNING] == [[float(i) for i in p["old_idx"]]]
    assert z._rows[0].params == p                  # no second computation


def test_sparkline_has_no_annotations_at_this_size(tk_root):
    z = _zone(tk_root)
    z.set_result(_result([_entry("QUIET")]))
    ax = z._rows[0].fig.axes[0]
    assert len(ax.texts) == 0                      # sentences belong on the big chart
    assert ax.get_xticks().size == 0 and ax.get_yticks().size == 0


def test_chronic_strip_lists_and_routes(tk_root):
    got = []
    z = _zone(tk_root, got)
    chronic = _entry("SICK")
    z.set_result(_result([_entry("HOT")], chronic=[chronic]))
    txt = " | ".join(_labels(z))
    assert "CHRONICALLY HIGH — stable, different problem (1)" in txt
    assert z._chronic_heading.winfo_manager() == "pack"
    assert "SICK" in txt and chronic.verdict in txt and chronic.sub_line in txt
    z._chronic_rows[0]._on_click()
    assert got == [("SICK", "linearity_fail_fraction")]


def test_chronic_strip_is_absent_when_nothing_is_chronic(tk_root):
    z = _zone(tk_root)
    z.set_result(_result([_entry("HOT")]))
    assert z._chronic_rows == []
    assert z._chronic_heading.winfo_manager() == ""
    assert z._chronic_body.winfo_manager() == ""
    assert "CHRONICALLY HIGH" not in " | ".join(_labels(z))


# ---- end-to-end: a REAL compute_focus_list row, rendered verbatim ---------

def _seed(db, model, n_lots=12, fails_last=0, start=D0, n_per=20, base_fails=2):
    from laser_trim_analyzer.database.models import (
        AnalysisResult, StatusType, SystemType)

    def add_lot(day, fails):
        with db.session() as s:
            for i in range(n_per):
                s.add(AnalysisResult(
                    model=model, serial=f"{model}-{day:%m%d}-{i}",
                    system=SystemType.A,          # NOT NULL; never read by SPC
                    filename=f"{model}_{i}_{day:%m-%d-%Y}.xls", file_date=day,
                    overall_status=(StatusType.FAIL if i < fails
                                    else StatusType.PASS)))

    for k in range(n_lots - 1):
        add_lot(start + timedelta(days=7 * k), base_fails)
    last = start + timedelta(days=7 * (n_lots - 1))
    add_lot(last, fails_last or base_fails)
    return last


def test_zone_renders_the_real_focus_row_verbatim(tk_root, tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.ml.spc import compute_focus_list
    db = DatabaseManager(tmp_path / "focus.db")
    _seed(db, "HOT", fails_last=12)
    res = compute_focus_list(db)
    assert res.focus, "seed must drift"
    z = _zone(tk_root)
    z.set_result(res, last_processed=res.anchor)
    texts = _labels(z)
    e = res.focus[0]
    assert e.verdict in texts and e.sub_line in texts   # producer's words, verbatim
    assert "HOT" in texts
    assert z._rows[0].entry is e                        # the same object the chart gets
