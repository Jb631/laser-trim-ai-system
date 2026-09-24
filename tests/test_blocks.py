"""The building blocks draw only from theme tokens and behave as the spec says (section 2)."""
import tkinter

import customtkinter as ctk
import pytest

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets import blocks


@pytest.fixture
def t(tk_root):
    return ThemeManager()


def _texts(w):
    out = []
    for c in w.winfo_children():
        if isinstance(c, (ctk.CTkLabel, ctk.CTkButton)):
            out.append(c.cget("text"))
        out.extend(_texts(c))
    return out


@pytest.mark.parametrize("verdict,fg,bg", [
    ("PASS", "PASS_FG", "PASS_BG"), ("fail", "FAIL_FG", "FAIL_BG"),
    ("UNTRIMMED", "NEUTRAL_FG", "NEUTRAL_BG"), ("NOT GRADED", "NEUTRAL_FG", "NEUTRAL_BG"),
    ("SIGMA WATCH", "WATCH_FG", "WATCH_BG"),
])
def test_every_verdict_carries_its_word_and_its_own_colour(tk_root, t, verdict, fg, bg):
    b = blocks.verdict_badge(tk_root, t, verdict)
    assert b.cget("text") == verdict.upper()
    assert b.cget("text_color") == getattr(t, fg) and b.cget("fg_color") == getattr(t, bg)


def test_an_unknown_verdict_is_neutral_never_a_pass_colour(tk_root, t):
    b = blocks.verdict_badge(tk_root, t, "SOMETHING NEW")
    assert b.cget("fg_color") == t.NEUTRAL_BG


def test_sigma_watch_is_never_drawn_as_a_failure(tk_root, t):
    # CLAUDE.md: sigma is a drift-watch signal, never a rejection.
    b = blocks.verdict_badge(tk_root, t, "SIGMA WATCH")
    assert b.cget("fg_color") != t.FAIL_BG and b.cget("text_color") != t.FAIL_FG


def test_pills_use_teal_to_act_and_coral_to_check(tk_root, t):
    assert blocks.count_pill(tk_root, t, 9).cget("text_color") == t.ACCENT
    c = blocks.count_pill(tk_root, t, 22, tone="check")
    assert c.cget("text_color") == t.CHECK and c.cget("text") == "22"
    assert blocks.count_pill(tk_root, t, 1234).cget("text") == "1,234"


def test_group_header_shows_title_count_unit_and_meaning(tk_root, t):
    g = blocks.group_header(tk_root, t, "Check the test", 22, column="tracks", tone="check",
                            meaning="Graded against more than one limit table")
    texts = _texts(g)
    for want in ("Check the test", "22", "tracks", "Graded against more than one limit table"):
        assert want in texts


def test_a_row_is_clickable_everywhere_and_lifts_on_hover(tk_root, t):
    hits = []
    r = blocks.row(tk_root, t, "6607", "Laser 1 (LTS): cut 6900 → try 6800", "~300",
                   tags=("both tracks",), on_click=lambda: hits.append(1))
    r.pack(); tk_root.update()
    assert {"6607", "Laser 1 (LTS): cut 6900 → try 6800", "~300", "both tracks"} <= set(_texts(r))
    r._on_click_all()                    # the handler bound to the row and every child
    assert hits == [1]
    r._set_hover(True);  assert r.cget("fg_color") == t.ELEVATED
    r._set_hover(False); assert r.cget("fg_color") == "transparent"


def test_a_real_click_fires_on_click_exactly_once_per_leaf(tk_root, t):
    """Review finding: CTk's bind() redirects a wrapper's binding onto ITS OWN internal real
    widgets (e.g. CTkLabel.bind() binds both its _canvas and its _label) -- and the old
    recursion into winfo_children() walked into those same internals again, so one real click
    fired on_click 2-4 times. r._on_click_all() (the other row test above) cannot see this: it
    calls the handler directly instead of letting Tk deliver an event through the (possibly
    stacked) bindings. This test drives REAL <Button-1> events at every underlying real Tk leaf.

    event_generate needs the widget genuinely viewable, which the withdrawn tk_root never is
    (confirmed empirically: event_generate dispatches nothing at all, even after update(), while
    withdrawn) -- so deiconify for the duration of the test and withdraw again after.

    Leaves are found with the BASE tkinter.Misc.winfo_children, not the public (possibly
    CTk-overridden) one: CTkFrame hides its own _canvas from the public version ("part of the
    frame itself"), which is correct for the production binding walk but would make a
    childless CTkFrame (the divider) look like a leaf itself here, when the real clickable
    surface a person's mouse actually lands on is the _canvas underneath it.
    """
    hits = []
    r = blocks.row(tk_root, t, "6607", "Laser 1 (LTS): cut 6900 → try 6800", "~300",
                   tags=("both tracks",), on_click=lambda: hits.append(1))
    r.pack()
    tk_root.deiconify()
    tk_root.update()
    try:
        def leaves(w):
            children = tkinter.Misc.winfo_children(w)
            if not children:
                yield w
            for c in children:
                yield from leaves(c)

        leaf_widgets = list(leaves(r))
        # 3 CTkFrames' _canvas (the row, mid, the divider) + 4 CTkLabels' _canvas+_label pairs
        # (model, statement, the one tag, value) = 11. A regression that stops binding
        # somewhere would also change this count, not just the per-leaf hit count below.
        assert len(leaf_widgets) == 11
        for leaf in leaf_widgets:
            hits.clear()
            leaf.event_generate("<Button-1>", x=1, y=1)
            tk_root.update()
            assert hits == [1], f"{leaf} fired on_click {len(hits)} times for one real click"
    finally:
        tk_root.withdraw()


def test_the_primary_button_carries_dark_text_on_teal(tk_root, t):
    b = blocks.primary_button(tk_root, t, "Open 6607", lambda: None)
    assert b.cget("fg_color") == t.ACCENT and b.cget("text_color") == t.TEXT_INVERSE


def test_a_check_banner_is_coral(tk_root, t):
    b = blocks.banner(tk_root, t, "Findings could not be loaded")
    assert b.cget("text_color") == t.CHECK and b.cget("fg_color") == t.CHECK_TINT


def test_no_block_hard_codes_a_colour():
    import inspect, re
    src = inspect.getsource(blocks)
    assert not re.search(r'"#[0-9a-fA-F]{6}"', src), "colours come from theme.py only"


def test_no_v6_heading_is_shouting():
    """Sentence case everywhere (spec section 2). Verdict words are labels, not headings."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"
    allowed = {"PASS", "FAIL", "UNTRIMMED", "NOT GRADED", "SIGMA WATCH"}
    shouting = []
    for p in root.rglob("*.py"):
        # Widened 2026-09-23 (fix round 1) to include digits and parentheses -- the original
        # class let "CUTS THE RECIPE DID NOT ASK FOR (LAST YEAR)" (findings_tab.py) slip past,
        # since the '(' broke the match before it ever reached the closing quote.
        for m in re.finditer(r'"([A-Z][A-Z0-9 \',&/()—-]{8,})"', p.read_text()):
            if m.group(1).strip() not in allowed:
                shouting.append(f"{p.name}: {m.group(1)}")
    assert not shouting, shouting
