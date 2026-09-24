"""Detector tests for scripts/render_pages.py's --audit clipped-text check (Task 9).

The detector's whole job is one comparison: is a text widget's ALLOCATED size
smaller than its REQUESTED (natural) size? That is real Tk geometry, not
something to fake with hand-picked numbers -- a fabricated (alloc, req) pair
would prove the ASSERTION, not the DETECTOR. So every test here builds a real
CustomTkinter scene and reads real winfo_width/winfo_reqwidth back.

Four things pinned first, each confirmed empirically against this repo's pinned
customtkinter (5.2.2) before being written (and, after the final review of
2026-09-24, the two ways of being cut off the size comparison cannot see -- a
label squeezed out entirely, a cell pushed past its container's edge -- plus the
walk reaching a tab view's tab names; see the section at the end):
  1. A label squeezed narrower than its text (a fixed-width frame with
     propagate off, exactly how a real column overflow happens) is reported.
  2. A label with room to spare is not.
  3. A widget that was never mapped (packed/gridded) at all -- e.g. a
     CTkTabview tab nobody has ever selected; CTkTabview only grids the
     active tab and never touches the rest -- is not reported either. Without
     this guard EVERY never-visited tab would read alloc=(1,1) against its
     full text's req and look like a clip that isn't there; verified by
     probing exactly this scene before the `winfo_ismapped()` guard existed.
  4. _open_first_row (the Findings page's opened-row guard) does not CLOSE a
     row that a previous render already re-opened -- the exact bug review
     found in the --audit driver itself (see test_a_row_still_open... below).

The first three run against an OFF-SCREEN-BUT-MAPPED root (see _offscreen
below), never a withdrawn one: a withdrawn CTk root leaves even a label with
room to spare at alloc=(1,1) (probed directly), which is the same ruling
render_pages.py's own module docstring documents for --audit. The fourth does
not need real geometry (it is a STATE bug, not a clipping one), so it runs
against a plain `tk_root`.
"""
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.render_pages import (_iter_widgets, _open_first_row,  # noqa: E402
                                  find_clipped_text_widgets, load_banner_shows)


def _offscreen(root, width=400, height=200):
    """Map `root` where nobody sees it: real geometry, never a window on the screen.

    Mirrors render_pages.py's own --audit driver (same ruling): deiconify at a
    position far outside any real display, THEN let Tk process the pending
    geometry requests. On macOS the window manager clamps +20000+20000 back into
    a corner of the screen (measured: rootx 1112 on a 1512-wide display), so the
    window is also made fully transparent -- still mapped and viewable, so the
    geometry is just as real.
    """
    try:
        root.attributes("-alpha", 0.0)
    except Exception:
        pass
    root.geometry(f"{width}x{height}+20000+20000")
    root.deiconify()
    root.update_idletasks()
    root.update()


def test_a_label_forced_narrower_than_its_text_is_reported(tk_root):
    import customtkinter as ctk

    _offscreen(tk_root)
    holder = ctk.CTkFrame(tk_root, width=30, height=20)
    holder.pack_propagate(False)          # the container refuses to grow for its child
    holder.pack()
    ctk.CTkLabel(holder, text="A label with much more text than 30 pixels can hold") \
        .pack(fill="both", expand=True)
    tk_root.update()

    found = find_clipped_text_widgets(tk_root, page="p", window_size="400x200")

    hits = [c for c in found if c.text.startswith("A label with much more text")]
    assert hits, f"expected the squeezed label to be reported; got {found}"
    hit = hits[0]
    assert hit.page == "p"
    assert hit.window_size == "400x200"
    assert hit.alloc[0] < hit.req[0], hit          # given less width than it asked for
    assert hit.req[0] - hit.alloc[0] > 1            # past the 1px rounding tolerance


def test_a_label_that_fits_is_not_reported(tk_root):
    import customtkinter as ctk

    _offscreen(tk_root)
    ctk.CTkLabel(tk_root, text="Fits fine").pack()
    tk_root.update()

    found = find_clipped_text_widgets(tk_root, page="p", window_size="400x200")

    assert not any("Fits fine" in c.text for c in found), found


def test_a_never_mapped_widget_is_not_reported(tk_root):
    """The CTkTabview trap: a tab nobody has selected yet is never gridded at
    all, so its children sit at Tk's un-placed default (probed: alloc=(1,1))
    forever, no matter how much text they hold. That is not a clip -- it is
    not on screen -- and must not be reported."""
    import customtkinter as ctk

    _offscreen(tk_root)
    holder = ctk.CTkFrame(tk_root, width=30, height=20)
    holder.pack_propagate(False)
    holder.pack()
    ctk.CTkLabel(holder, text="Never placed anywhere, would clip if it were")
    # deliberately never .pack()/.grid()'d on `holder` or `tk_root`
    tk_root.update()

    found = find_clipped_text_widgets(tk_root, page="p", window_size="400x200")

    assert not any("Never placed" in c.text for c in found), found


def _one_finding_row():
    """The minimal raw finding dict findings.presentation.arrange() turns into
    ONE real row (not a special-cased "other"/cut_setting one -- just enough
    to reach _draw_detail with a summary and no settings table)."""
    return {"model": "TESTMODEL", "analyzer": "limit_tables",
            "title": "Test finding for the opened-row guard",
            "summary": "A short summary for the detail pane.",
            "evidence": {}, "n_units": 5}


def test_a_row_still_open_from_a_previous_render_is_not_closed(tk_root):
    """Review finding (2026-09-24): FindingsView._render() re-opens whatever
    row was open before a refresh (`was_open` -> toggle(was_open)), and the
    Findings PAGE is shown more than once in one --audit run. By the second
    window size, the row _open_first_row opened for the first size is
    routinely already open again by the time this runs -- and an unconditional
    toggle() on an ALREADY-open key CLOSES it, which is exactly the bug: drive
    a real FindingsView through that same two-render sequence and confirm the
    row ends OPEN, not closed.
    """
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.findings_view import FindingsView

    finding = _one_finding_row()
    view = FindingsView(tk_root, ThemeManager(), include_empty=False)

    view.set_findings([finding])                       # first render: open_key starts None
    assert _open_first_row(view) is True
    key = next(iter(view.row_widgets))
    assert view.open_key == key and view._detail is not None

    view.set_findings([finding])                        # second render == a reload
    # Sanity on the test's own premise: _render()'s "keep the open row open
    # across a refresh" must have already re-opened it by itself.
    assert view.open_key == key and view._detail is not None, (
        "test setup assumption broken -- FindingsView._render() did not "
        "re-open the row on its own; _open_first_row was never exercised "
        "against the state the real bug needs")

    assert _open_first_row(view) is True, "a row already open must stay open"
    assert view.open_key == key
    assert view._detail is not None


# ---- What the audit used to miss (final review, 2026-09-24) ---------------------------------------
# A label squeezed out ENTIRELY is unmapped by its geometry manager, so the old "skip what is not
# mapped" rule skipped it; a grid cell pushed past its container keeps its full size, so the size
# comparison passed it. Both are cut off. Each test below was run against the old detector first.

def test_a_label_squeezed_out_entirely_is_reported(tk_root):
    import customtkinter as ctk

    _offscreen(tk_root)
    holder = ctk.CTkFrame(tk_root, width=60, height=30)
    holder.pack_propagate(False)
    holder.pack()
    ctk.CTkLabel(holder, text="Takes all the room there is").pack(side="left")
    ctk.CTkLabel(holder, text="Squeezed out").pack(side="left")     # no room left: unmapped
    tk_root.update()

    hits = [c for c in find_clipped_text_widgets(tk_root, page="p") if c.text == "Squeezed out"]
    assert hits, "a label given no room at all must be reported, not skipped as off-screen"
    assert hits[0].why.startswith("squeezed out")


def test_a_grid_cell_pushed_past_its_containers_edge_is_reported(tk_root):
    import customtkinter as ctk

    _offscreen(tk_root)
    holder = ctk.CTkFrame(tk_root, width=120, height=40)
    holder.grid_propagate(False)
    holder.pack()
    ctk.CTkLabel(holder, text="A wide first column", width=150).grid(row=0, column=0)
    ctk.CTkLabel(holder, text="Pushed out").grid(row=0, column=1)   # starts beyond the edge
    tk_root.update()

    hits = [c for c in find_clipped_text_widgets(tk_root, page="p") if c.text == "Pushed out"]
    assert hits, "a label beyond its container's right edge must be reported"
    assert hits[0].why.startswith("past the right edge")
    assert hits[0].alloc[0] >= hits[0].req[0]      # full size: the size comparison alone passes it


def test_a_label_on_a_tab_nobody_has_selected_is_not_reported(tk_root):
    """Unmapped because nothing LAID OUT its tab (CTkTabview grid_forgets the others) -- hidden,
    not squeezed out. Checked both for a tab never selected and one selected and left again."""
    import customtkinter as ctk

    _offscreen(tk_root)
    tv = ctk.CTkTabview(tk_root, width=380, height=150)
    tv.pack()
    tv.add("One")
    tv.add("Two")
    tv.add("Three")
    ctk.CTkLabel(tv.tab("Two"), text="On a tab never selected").pack()
    ctk.CTkLabel(tv.tab("Three"), text="On a tab selected and left").pack()
    tk_root.update()
    tv.set("Three")
    tk_root.update()
    tv.set("One")
    tk_root.after(300)                   # CTkTabview grid_forgets the old tab 100 ms after set()
    tk_root.update()

    found = find_clipped_text_widgets(tk_root, page="p")
    assert not any(c.text.startswith("On a tab") for c in found), found


def test_the_walk_reaches_every_tab_name(tk_root):
    """CTkTabview.winfo_children() hides its segmented button -- its tab names -- so a walk through
    the public method never audited the Model page's seven tab names at all."""
    import tkinter
    import customtkinter as ctk

    tv = ctk.CTkTabview(tk_root)
    tv.add("Drift Metrics")
    tv.add("Final Test Units")
    texts = {w.cget("text") for w in _iter_widgets(tv) if isinstance(w, tkinter.Label)}
    assert {"Drift Metrics", "Final Test Units"} <= texts


def test_the_forced_failure_pass_only_counts_a_banner_showing_what_failed(tk_root):
    import customtkinter as ctk

    class Page:
        pass

    page = Page()
    page._load_banner = ctk.CTkLabel(tk_root, text="")
    assert not load_banner_shows(page, "unit list")                  # never laid out, no text
    page._load_banner.configure(text="⚠ Could not load: unit list. Those parts of this page…")
    assert not load_banner_shows(page, "unit list")                  # the right words, not shown
    page._load_banner.pack()
    assert load_banner_shows(page, "unit list")
    page._load_banner.configure(text="⚠ Could not load: lot chart.")
    assert not load_banner_shows(page, "unit list")                  # shown, but not what was forced
    assert not load_banner_shows(Page(), "unit list")                # no banner at all


def test_the_audit_app_has_the_saved_settings_and_can_never_save_them(tmp_path, monkeypatch):
    """Config() defaults audited everything that depends on James's settings EMPTY (the Home
    page's folder list...). Config.load() fixes that -- and makes a save dangerous: it would write
    his real config.yaml pointing at a throwaway copy. So the settings load, and saving refuses."""
    import pytest
    from laser_trim_analyzer import config as cfgmod
    import laser_trim_analyzer.database as dbpkg
    from laser_trim_analyzer.database import manager as mgr
    from scripts import render_pages

    # _build_app sets both database globals; put them back after this test.
    monkeypatch.setattr(mgr, "_db_manager", mgr._db_manager)
    monkeypatch.setattr(dbpkg, "_db_manager", getattr(dbpkg, "_db_manager", None), raising=False)
    saved = cfgmod.Config()
    saved.ingest.folders = ["/invented/laser-one-share", "/invented/final-test-share"]
    saved.save()                           # into the conftest's tmp app folder, never the real one
    config_file = cfgmod.get_app_directory() / "data" / "config.yaml"
    before = config_file.read_text()

    app, db = render_pages._build_app(tmp_path / "qa_copy.db")
    app.withdraw()
    try:
        assert app.config.ingest.folders == ["/invented/laser-one-share", "/invented/final-test-share"]
        assert pathlib.Path(app.config.database.path) == tmp_path / "qa_copy.db"
        with pytest.raises(RuntimeError, match="never saved"):
            app.config.save()
        assert config_file.read_text() == before
    finally:
        app.destroy()
        db.close()
