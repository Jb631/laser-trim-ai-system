"""The building blocks draw only from theme tokens and behave as the spec says (section 2)."""
import re
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
    withdrawn) -- so deiconify for the duration of the test, invisibly, and withdraw again after.

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
    # Mapped (event_generate needs it) but never seen during the gate: macOS clamps
    # +20000+20000 back into a screen corner, so the window is also fully transparent.
    try:
        tk_root.attributes("-alpha", 0.0)
    except Exception:
        pass
    tk_root.geometry("+20000+20000")
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


def test_leaving_a_row_for_another_row_whose_name_starts_the_same_drops_the_hover(tk_root, t):
    """Tk names siblings .!ctkframe, .!ctkframe2 ... .!ctkframe20. The leave check asked whether
    the widget under the pointer had a path STARTING with the row's -- and ".!ctkframe20.!ctklabel"
    starts with ".!ctkframe2", so moving from the 2nd row onto the 20th left the 2nd lit."""
    holder = ctk.CTkFrame(tk_root)
    rows = [blocks.row(holder, t, f"M{i}", f"statement {i}", "1", on_click=lambda: None)
            for i in range(1, 21)]
    second, twentieth = rows[1], rows[19]
    assert str(second).endswith("!ctkframe2") and str(twentieth).endswith("!ctkframe20")
    inside_twentieth = twentieth.winfo_children()[0]            # its model label

    class Event:
        x_root = y_root = 0

    second._set_hover(True)
    second.winfo_containing = lambda x, y: inside_twentieth     # the pointer is on row 20 now
    second._on_leave(Event())
    assert second.cget("fg_color") == "transparent"

    second._set_hover(True)
    second.winfo_containing = lambda x, y: second.winfo_children()[0]   # still on row 2's own label
    second._on_leave(Event())
    assert second.cget("fg_color") == t.ELEVATED                 # a child is still inside


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


# ---- wrap_to_width ---------------------------------------------------------------------------
# <Configure> never fires under the withdrawn tk_root, even though geometry propagation still
# runs (winfo_width() reports the real number) -- confirmed empirically the same way
# test_a_real_click_fires_on_click_exactly_once_per_leaf above did for event_generate: map the
# window off-screen (alpha 0, +20000+20000) so real <Configure> events are actually dispatched.

def _mapped_offscreen(root):
    try:
        root.attributes("-alpha", 0.0)
    except Exception:
        pass
    root.geometry("+20000+20000")
    root.deiconify()
    root.update_idletasks()
    root.update()


def test_wrap_to_width_sets_wraplength_from_the_container_after_update(tk_root, t):
    container = ctk.CTkFrame(tk_root, fg_color="transparent")
    container.pack(fill="both", expand=True)
    label = ctk.CTkLabel(container, text="x" * 300)
    label.pack(fill="x")
    tk_root.geometry("400x200")
    _mapped_offscreen(tk_root)
    try:
        blocks.wrap_to_width(label, container, padding=20)
        assert container.winfo_width() == 400          # the container really is 400 wide
        assert label.cget("wraplength") == 400 - 20      # set once, immediately
    finally:
        tk_root.withdraw()


def test_wrap_to_width_follows_the_container_when_it_resizes(tk_root, t):
    container = ctk.CTkFrame(tk_root, fg_color="transparent")
    container.pack(fill="both", expand=True)
    label = ctk.CTkLabel(container, text="x" * 300)
    label.pack(fill="x")
    tk_root.geometry("400x200")
    _mapped_offscreen(tk_root)
    try:
        blocks.wrap_to_width(label, container, padding=20)
        tk_root.geometry("300x200")
        tk_root.update_idletasks()
        tk_root.update()
        assert label.cget("wraplength") == 300 - 20
    finally:
        tk_root.withdraw()


def test_wrap_to_width_never_drops_below_120(tk_root, t):
    container = ctk.CTkFrame(tk_root, fg_color="transparent")
    container.pack(fill="both", expand=True)
    label = ctk.CTkLabel(container, text="x")
    label.pack(fill="x")
    tk_root.geometry("140x100")           # 140 - 40 padding would be 100, below the floor
    _mapped_offscreen(tk_root)
    try:
        blocks.wrap_to_width(label, container, padding=40)
        assert label.cget("wraplength") == 120
    finally:
        tk_root.withdraw()


def test_wrap_to_width_defaults_padding_to_zero(tk_root, t):
    container = ctk.CTkFrame(tk_root, fg_color="transparent")
    container.pack(fill="both", expand=True)
    label = ctk.CTkLabel(container, text="x" * 50)
    label.pack(fill="x")
    tk_root.geometry("500x100")
    _mapped_offscreen(tk_root)
    try:
        blocks.wrap_to_width(label, container)
        assert label.cget("wraplength") == 500
    finally:
        tk_root.withdraw()


def test_wrap_to_width_never_replaces_an_existing_configure_handler(tk_root, t):
    """The container may already have its own <Configure> binding (a chart redraw, another
    wrap) -- wrap_to_width must ADD to it, never replace it (global-constraints.md, Task 1).

    A plain tkinter.Frame, not a CTkFrame: CTkFrame.bind() (see blocks.row()'s own docstring)
    hard-codes add=True on the Tk call underneath REGARDLESS of what its caller passes, so a
    CTkFrame container could never actually catch a dropped add="+" here -- only a widget whose
    native .bind() defaults to replace (confirmed empirically: without add="+", a second
    tkinter.Frame.bind() call drops the first handler entirely) can prove this contract.
    """
    container = tkinter.Frame(tk_root)
    container.pack(fill="both", expand=True)
    label = ctk.CTkLabel(container, text="x" * 10)
    label.pack(fill="x")
    hits = []
    container.bind("<Configure>", lambda e: hits.append(1))
    tk_root.geometry("400x200")
    _mapped_offscreen(tk_root)
    try:
        blocks.wrap_to_width(label, container, padding=0)
        hits.clear()
        tk_root.geometry("300x200")
        tk_root.update_idletasks()
        tk_root.update()
        assert hits, "wrap_to_width replaced the pre-existing <Configure> handler instead of adding to it"
        assert label.cget("wraplength") == 300
    finally:
        tk_root.withdraw()


_SHOUT = re.compile(r"^\s*([A-Z][A-Z0-9]*(?:[- ,&/'—]+[A-Z][A-Z0-9]*)*)(?![a-z])")
_VERDICT_LABELS = {"PASS", "FAIL", "UNTRIMMED", "NOT GRADED", "SIGMA WATCH"}


def _literal_starts(source: str):
    """(line, text) for the START of every string a user could read: a plain literal's whole
    text, an f-string's text up to its first replacement field. Two kinds of literal are not
    starts: the continuation of an implicit concatenation ("..." "...", mid-sentence), and a
    literal that is a whole statement (a docstring -- never on screen)."""
    import io
    import tokenize

    starts = []
    skip = {tokenize.NL, tokenize.COMMENT}
    statement_start = {tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}
    fstart = getattr(tokenize, "FSTRING_START", None)          # Python 3.12+ splits f-strings up
    fmiddle = getattr(tokenize, "FSTRING_MIDDLE", None)
    fend = getattr(tokenize, "FSTRING_END", None)
    prev = None                  # type of the previous significant token
    depth = 0                    # inside an f-string (3.12+ tokens)
    take_next_middle = False
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in skip:
            continue
        if depth:
            if tok.type == fmiddle and take_next_middle:
                starts.append((tok.start[0], tok.string))
            take_next_middle = False
            if tok.type == fstart:
                depth += 1
            elif tok.type == fend:
                depth -= 1
                if not depth:
                    prev = tok.type
            continue
        is_string = tok.type == tokenize.STRING or tok.type == fstart
        if is_string and prev not in (tokenize.STRING, fend) and prev not in statement_start \
                and prev is not None:
            if tok.type == fstart:
                take_next_middle = True
            else:
                m = re.match(r"(?is)^([rbuf]*)('\'\'|\"\"\"|'|\")(.*)\2$", tok.string)
                if m:
                    text = m.group(3)
                    if "f" in m.group(1).lower():                 # a pre-3.12 f-string token
                        text = re.split(r"(?<!\{)\{(?!\{)", text)[0]
                    starts.append((tok.start[0], text))
        if tok.type == fstart:
            depth = 1
            continue
        prev = tok.type
    return starts


def test_no_v6_heading_is_shouting():
    """Sentence case everywhere (spec section 2). Verdict words are labels, not headings.

    Widened twice. 2026-09-23: digits and parentheses (a '(' used to break the match, so
    "CUTS THE RECIPE DID NOT ASK FOR (LAST YEAR)" slipped past). 2026-09-24 (final review): the old
    scan matched only a string that was ALL capitals from quote to quote, so a heading that SHOUTS
    and then goes on in lower case passed it -- f"CHRONICALLY HIGH — stable, different problem
    ({n})" and "LIN-PASSING (accepted)". It now reads the START of every literal, f-strings
    included, and flags a leading run of capitals eight letters or longer.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"
    shouting, seen = [], 0
    for p in root.rglob("*.py"):
        for line, text in _literal_starts(p.read_text()):
            seen += 1
            m = _SHOUT.match(text)
            if not m:
                continue
            run = m.group(1).strip(" ,&/'—-")
            if run not in _VERDICT_LABELS and sum(c.isalpha() for c in run) >= 8:
                shouting.append(f"{p.name}:{line}: {text[:60]!r}")
    assert seen > 500, f"the scan read only {seen} string starts -- it is not reading the source"
    assert not shouting, shouting


def test_the_shouting_scan_reads_f_strings_and_skips_what_nobody_sees():
    src = ('"""A DOCSTRING THAT SHOUTS is never on screen."""\n'
           'x = f"CHRONICALLY HIGH — stable ({n})"\n'
           'y = ("Items not recognised (add-ons such as "\n     "FAI/LAT/TEST UNITS, and more)")\n'
           'z = "LIN-PASSING (accepted)"\n')
    starts = [text for _line, text in _literal_starts(src)]
    assert any(t.startswith("CHRONICALLY HIGH") for t in starts)       # an f-string's start
    assert "LIN-PASSING (accepted)" in starts
    assert not any(t.startswith("A DOCSTRING") for t in starts)         # a docstring
    assert not any(t.startswith("FAI/LAT") for t in starts)             # mid-sentence continuation
