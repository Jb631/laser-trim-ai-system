"""The shared findings view: rows per group, one open at a time, Show all, open the model."""
import customtkinter as ctk

from laser_trim_analyzer.findings import presentation as P
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets.findings_view import FindingsView


def cut(model, tpy, best=6800.0, current=6900.0, track="Track A"):
    return {"analyzer": "cut_setting", "model": model, "category": "Cut setting", "title": "t",
            "summary": f"summary for {model} {track}", "systems": ["B"], "n_units": 100,
            "tracks_per_year": tpy,
            "evidence": {"best": best, "current": current, "grade": "two_periods", "track": track,
                         "group": {"settings": [
                             {"setting": best, "n": 813, "pass_pct": 87.1, "window": "a .. b"},
                             {"setting": current, "n": 1283, "pass_pct": 60.4, "window": "c .. d"}]}}}


def cut_track(model, tpy, track, settings, best=6800.0, current=6900.0):
    """Like cut(), but with an explicit per-track settings list -- for testing how a merged
    row's settings table combines two tracks (join by setting value, not list position)."""
    return {"analyzer": "cut_setting", "model": model, "category": "Cut setting", "title": "t",
            "summary": f"summary for {model} {track}", "systems": ["B"], "n_units": 100,
            "tracks_per_year": tpy,
            "evidence": {"best": best, "current": current, "grade": "two_periods", "track": track,
                         "group": {"settings": settings}}}


def _texts(w):
    out = []
    for c in w.winfo_children():
        if isinstance(c, (ctk.CTkLabel, ctk.CTkButton)):
            out.append(c.cget("text"))
        out.extend(_texts(c))
    return out


def test_each_group_shows_five_rows_then_offers_the_rest(tk_root):
    v = FindingsView(tk_root, ThemeManager())
    v.set_findings([cut(f"M{i}", float(100 - i), best=float(i)) for i in range(8)])
    assert len(v.row_widgets) == P.ROWS_PER_GROUP
    assert "Show all 8" in _texts(v)
    v.show_all("yield")
    assert len(v.row_widgets) == 8 and "Show all 8" not in _texts(v)


def test_opening_a_row_shows_its_evidence_and_only_one_is_open(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None)
    v.set_findings([cut("6607", 182.0), cut("8232-1", 66.0, best=4100.0, current=4000.0)])
    first, second = list(v.row_widgets)
    v.toggle(first)
    assert v.open_key == first and any("summary for 6607" in x for x in _texts(v))
    assert "Open 6607" in _texts(v)
    v.toggle(second)
    assert v.open_key == second and not any("summary for 6607" in x for x in _texts(v))
    v.toggle(second)
    assert v.open_key is None


def test_the_open_button_goes_to_the_model(tk_root):
    opened = []
    v = FindingsView(tk_root, ThemeManager(), on_open=opened.append)
    v.set_findings([cut("6607", 182.0)])
    v.toggle(next(iter(v.row_widgets)))
    btn = [b for b in v._detail.winfo_children() if isinstance(b, ctk.CTkButton)][0]
    btn.invoke()
    assert opened == ["6607"]


def test_without_an_open_handler_there_is_no_open_button(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=None)     # the model page's own tab
    v.set_findings([cut("6607", 182.0)])
    v.toggle(next(iter(v.row_widgets)))
    assert not any(x.startswith("Open ") for x in _texts(v))


def test_the_open_button_is_teal_filled_by_default(tk_root):
    """The Findings page's own default (open_as="primary", unchanged) -- opening a row IS the
    page's one call to action."""
    t = ThemeManager()
    v = FindingsView(tk_root, t, on_open=lambda m: None)
    v.set_findings([cut("6607", 182.0)])
    v.toggle(next(iter(v.row_widgets)))
    btn = [b for b in v._detail.winfo_children() if isinstance(b, ctk.CTkButton)][0]
    assert btn.cget("fg_color") == t.ACCENT


def test_open_as_link_draws_a_link_not_a_second_teal_button(tk_root):
    """Home (facelift step 2 review, 7bc0743): Home already has its own primary_button
    ("Process everything new"), so its embedded FindingsView passes open_as="link" -- expanding
    a row must never draw a SECOND teal-filled button (global-constraints.md: at most ONE per
    screen). Red before FindingsView grew this option: every open button was primary_button."""
    t = ThemeManager()
    opened = []
    v = FindingsView(tk_root, t, on_open=opened.append, open_as="link")
    v.set_findings([cut("6607", 182.0)])
    v.toggle(next(iter(v.row_widgets)))
    btn = [b for b in v._detail.winfo_children() if isinstance(b, ctk.CTkButton)][0]
    assert btn.cget("text") == "Open 6607"
    assert btn.cget("fg_color") != t.ACCENT
    btn.invoke()
    assert opened == ["6607"]


def test_a_merged_row_opens_with_each_track_named(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None)
    v.set_findings([cut("6607", 182.0, track="Track A"), cut("6607", 118.0, track="Track B")])
    assert len(v.row_widgets) == 1
    v.toggle(next(iter(v.row_widgets)))
    texts = _texts(v)
    # Both tracks are named in ONE line (the order the table's joined cells use below it),
    # not a separate heading per track -- see test_a_merged_row_draws_one_settings_table_*.
    assert "Track A · Track B" in texts and "~300" in texts


def test_a_merged_row_draws_one_settings_table_with_joined_cells(tk_root):
    """Spec section 3 / the approved mockup: a merged row's settings table is ONE table --
    setting, tracks, in spec, no 'ran' column -- whose cells join both tracks' numbers with
    ' · ' (the mockup's real example: 813 · 766, 87% · 89%)."""
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None)
    v.set_findings([
        cut_track("6607", 182.0, "Track A",
                  [{"setting": 6800.0, "n": 813, "pass_pct": 87.1},
                   {"setting": 6900.0, "n": 1283, "pass_pct": 60.4}]),
        cut_track("6607", 118.0, "Track B",
                  [{"setting": 6800.0, "n": 766, "pass_pct": 89.0},
                   {"setting": 6900.0, "n": 1207, "pass_pct": 70.0}]),
    ])
    v.toggle(next(iter(v.row_widgets)))
    texts = _texts(v)
    assert "813 · 766" in texts and "1,283 · 1,207" in texts
    assert "87% · 89%" in texts and "60% · 70%" in texts
    assert texts.count("cut") == 1                 # exactly ONE table, not one per track
    assert "ran" not in texts                       # the window column is gone


def test_a_merged_rows_settings_line_up_by_setting_not_by_list_position(tk_root):
    """A naive positional zip would cross-match Track B's 6900 row with Track A's 6800 numbers
    here, since Track B lists its two settings in the OPPOSITE order."""
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None)
    v.set_findings([
        cut_track("6607", 182.0, "Track A",
                  [{"setting": 6800.0, "n": 813, "pass_pct": 87.1},
                   {"setting": 6900.0, "n": 1283, "pass_pct": 60.4}]),
        cut_track("6607", 118.0, "Track B",
                  [{"setting": 6900.0, "n": 1207, "pass_pct": 70.0},
                   {"setting": 6800.0, "n": 766, "pass_pct": 89.0}]),
    ])
    v.toggle(next(iter(v.row_widgets)))
    texts = _texts(v)
    assert "813 · 766" in texts and "1,283 · 1,207" in texts


def test_a_merged_rows_track_missing_a_setting_shows_a_dash(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None)
    v.set_findings([
        cut_track("6607", 182.0, "Track A",
                  [{"setting": 6800.0, "n": 813, "pass_pct": 87.1},
                   {"setting": 6900.0, "n": 1283, "pass_pct": 60.4}]),
        cut_track("6607", 118.0, "Track B",
                  [{"setting": 6800.0, "n": 766, "pass_pct": 89.0}]),      # never ran 6900
    ])
    v.toggle(next(iter(v.row_widgets)))
    texts = _texts(v)
    assert "1,283 · —" in texts and "60% · —" in texts


def test_an_empty_group_says_what_would_fill_it(tk_root):
    v = FindingsView(tk_root, ThemeManager(), include_empty=True)
    v.set_findings([cut("6607", 182.0)])
    texts = _texts(v)
    assert any(x.startswith("No recipe or setting changes found") for x in texts)


def test_the_tab_view_hides_empty_groups(tk_root):
    v = FindingsView(tk_root, ThemeManager(), include_empty=False)
    v.set_findings([cut("6607", 182.0)])
    assert "What changed" not in _texts(v)


# ---- rows_per_group / groups (Task 1) --------------------------------------------------------

def test_rows_per_group_overrides_the_default_row_count(tk_root):
    v = FindingsView(tk_root, ThemeManager(), rows_per_group=3)
    v.set_findings([cut(f"M{i}", float(100 - i), best=float(i)) for i in range(8)])
    assert len(v.row_widgets) == 3
    assert "Show all 8" in _texts(v)


def test_rows_per_group_default_is_unchanged(tk_root):
    """No argument -> presentation.ROWS_PER_GROUP, exactly as before this option existed."""
    v = FindingsView(tk_root, ThemeManager())
    v.set_findings([cut(f"M{i}", float(100 - i), best=float(i)) for i in range(8)])
    assert len(v.row_widgets) == P.ROWS_PER_GROUP


def test_groups_limits_which_groups_are_drawn(tk_root):
    v = FindingsView(tk_root, ThemeManager(), groups=("yield",), include_empty=True)
    v.set_findings([cut("6607", 182.0), _history(58.0, 80.0)])
    texts = _texts(v)
    assert "Change a setting to raise yield" in texts
    assert "What changed" not in texts               # history group never drawn, empty or not
    assert "Check the test" not in texts
    assert "Laser time you could save" not in texts


def test_groups_none_draws_every_group_exactly_as_before(tk_root):
    v = FindingsView(tk_root, ThemeManager(), groups=None, include_empty=True)
    v.set_findings([cut("6607", 182.0)])
    texts = _texts(v)
    for title in ("Change a setting to raise yield", "Laser time you could save",
                  "Check the test", "What changed"):
        assert title in texts


def test_groups_can_admit_the_other_group_only_when_listed(tk_root):
    unmapped = {"analyzer": "a_future_analyzer", "model": "6607", "title": "t", "summary": "s",
                "systems": ["B"], "n_units": 10, "evidence": {}}
    v = FindingsView(tk_root, ThemeManager(), groups=("yield",), include_empty=False)
    v.set_findings([cut("6607", 182.0), unmapped])
    assert "Other findings" not in _texts(v)          # not listed -> never drawn, even with rows

    v2 = FindingsView(tk_root, ThemeManager(), groups=("yield", "other"), include_empty=False)
    v2.set_findings([cut("6607", 182.0), unmapped])
    assert "Other findings" in _texts(v2)


# ---- Each row opens ITS OWN evidence (final review, 2026-09-24) ----------------------------------

def lt(track):
    """limit_tables writes one finding per (laser, track), and its title names neither."""
    return {"analyzer": "limit_tables", "model": "6607", "category": "Limit table",
            "title": "Laser 1 (LTS): the limit table changed", "summary": f"the evidence for {track}",
            "systems": ["B"], "n_units": 100, "evidence": {"track": track}}


def _invisible_but_mapped(root):
    """Tk delivers a real event only to a VIEWABLE window -- event_generate dispatches nothing
    under the withdrawn tk_root -- so map it, but where nobody sees it: fully transparent, and
    far off-screen where the window manager allows it (macOS clamps +20000+20000 back into a
    corner of the screen; alpha 0 is what hides it there)."""
    try:
        root.attributes("-alpha", 0.0)
    except Exception:
        pass
    root.geometry("1100x700+20000+20000")
    root.deiconify()
    root.update_idletasks()
    root.update()


def _click(row):
    """A REAL <Button-1> on the statement label's inner Tk label -- dispatched by Tk through the
    bindings blocks.row() installed, never by calling view.toggle(key) or a test hook."""
    import tkinter

    def leaves(w):
        kids = tkinter.Misc.winfo_children(w)
        if not kids:
            yield w
        for c in kids:
            yield from leaves(c)
    target = [w for w in leaves(row) if isinstance(w, tkinter.Label)
              and w.cget("text").startswith("Laser 1 (LTS)")][0]
    target.event_generate("<Button-1>", x=2, y=2)
    target.update()


def _opened_summary(view):
    return [c.cget("text") for c in view._detail.winfo_children() if isinstance(c, ctk.CTkLabel)][0]


def test_each_of_two_rows_that_read_alike_opens_its_own_evidence_on_a_real_click(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None, include_empty=False)
    v.pack(fill="both", expand=True)
    v.set_findings([lt("Track A"), lt("Track B")])
    _invisible_but_mapped(tk_root)
    try:
        assert len(v.row_widgets) == 2
        by_track = {v._rows[k].findings[0]["evidence"]["track"]: w for k, w in v.row_widgets.items()}
        for track in ("Track A", "Track B", "Track A"):
            row = by_track[track]
            _click(row)
            assert v._detail is not None, f"clicking {track}'s row opened nothing"
            assert _opened_summary(v) == f"the evidence for {track}"
            order = v.pack_slaves()
            assert order.index(v._detail) == order.index(row) + 1, "the detail opened under another row"
    finally:
        tk_root.withdraw()


def test_the_open_row_stays_the_same_track_when_a_refresh_reorders_the_findings(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None, include_empty=False)
    v.set_findings([lt("Track A"), lt("Track B")])
    key_a = [k for k, r in v._rows.items() if r.findings[0]["evidence"]["track"] == "Track A"][0]
    v.toggle(key_a)
    v.set_findings([lt("Track B"), lt("Track A")])        # the same findings, the other order
    assert _opened_summary(v) == "the evidence for Track A"


def _history(pct_before, pct_after, **tables):
    return {"analyzer": "recipe_change", "model": "7715", "category": "Setting change",
            "title": "Laser 1 (LTS): recipe changed from 1 cut (cut length 2950) to 2 cuts (cut length 2950, 4500)",
            "summary": "s", "systems": ["B"], "n_units": 400,
            "evidence": {"before": {"trim_pass_pct": pct_before},
                         "after": {"trim_pass_pct": pct_after, "first": "2025-01-13"}, **tables}}


def test_a_move_across_a_table_change_is_drawn_without_colour_and_says_different_test(tk_root):
    t = ThemeManager()
    v = FindingsView(tk_root, t, include_empty=False)
    v.set_findings([_history(58.0, 80.0, limit_table_changed=True, limit_tables_mixed=True)])
    row = next(iter(v.row_widgets.values()))
    readout = [w for w in row.winfo_children() if isinstance(w, ctk.CTkLabel) and w.cget("text") == "+22"][0]
    assert readout.cget("text_color") == t.TEXT_PRIMARY          # neither PASS green nor coral
    assert "different test" in _texts(row)


def test_a_like_for_like_move_is_still_green_or_coral(tk_root):
    t = ThemeManager()
    v = FindingsView(tk_root, t, include_empty=False)
    v.set_findings([_history(58.0, 80.0, limit_table_changed=False, limit_tables_mixed=False)])
    row = next(iter(v.row_widgets.values()))
    readout = [w for w in row.winfo_children() if isinstance(w, ctk.CTkLabel) and w.cget("text") == "+22"][0]
    assert readout.cget("text_color") == t.PASS_FG and "different test" not in _texts(row)


def test_a_rework_rows_readout_says_unit_days_beside_a_rows_in_tracks(tk_root):
    """One group, two units: the laser-time column counts tracks, rework load counts unit-days --
    the rework row's readout says so, the other row's stays a bare count."""
    rework = {"analyzer": "rework_load", "model": "R1", "category": "Rework load",
              "title": "Laser 1 (LTS): 261 units a year fail here and pass final test after rework",
              "summary": "s", "systems": ["B"], "n_units": 261,
              "evidence": {"facts": {"rework_unit_days": 261}}}
    effort = {"analyzer": "trim_effort", "model": "E1", "category": "Trim avoidance",
              "title": "t", "summary": "s", "systems": ["B"], "n_units": 1008,
              "evidence": {"facts": {"arrive_in_spec_n": 1008}}}
    v = FindingsView(tk_root, ThemeManager(), include_empty=False)
    v.set_findings([rework, effort])
    texts = _texts(v)
    assert "261 unit-days" in texts and "1,008" in texts

