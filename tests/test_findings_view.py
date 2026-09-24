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


def test_a_merged_row_opens_with_each_track_named(tk_root):
    v = FindingsView(tk_root, ThemeManager(), on_open=lambda m: None)
    v.set_findings([cut("6607", 182.0, track="Track A"), cut("6607", 118.0, track="Track B")])
    assert len(v.row_widgets) == 1
    v.toggle(next(iter(v.row_widgets)))
    texts = _texts(v)
    assert "Track A" in texts and "Track B" in texts and "~300" in texts


def test_an_empty_group_says_what_would_fill_it(tk_root):
    v = FindingsView(tk_root, ThemeManager(), include_empty=True)
    v.set_findings([cut("6607", 182.0)])
    texts = _texts(v)
    assert any(x.startswith("No recipe changes found") for x in texts)


def test_the_tab_view_hides_empty_groups(tk_root):
    v = FindingsView(tk_root, ThemeManager(), include_empty=False)
    v.set_findings([cut("6607", 182.0)])
    assert "What changed" not in _texts(v)
