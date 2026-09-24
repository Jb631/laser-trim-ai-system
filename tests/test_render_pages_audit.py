"""Detector tests for scripts/render_pages.py's --audit clipped-text check (Task 9).

The detector's whole job is one comparison: is a text widget's ALLOCATED size
smaller than its REQUESTED (natural) size? That is real Tk geometry, not
something to fake with hand-picked numbers -- a fabricated (alloc, req) pair
would prove the ASSERTION, not the DETECTOR. So every test here builds a real
CustomTkinter scene and reads real winfo_width/winfo_reqwidth back.

Three things pinned, each confirmed empirically against this repo's pinned
customtkinter (5.2.2) before being written:
  1. A label squeezed narrower than its text (a fixed-width frame with
     propagate off, exactly how a real column overflow happens) is reported.
  2. A label with room to spare is not.
  3. A widget that was never mapped (packed/gridded) at all -- e.g. a
     CTkTabview tab nobody has ever selected; CTkTabview only grids the
     active tab and never touches the rest -- is not reported either. Without
     this guard EVERY never-visited tab would read alloc=(1,1) against its
     full text's req and look like a clip that isn't there; verified by
     probing exactly this scene before the `winfo_ismapped()` guard existed.

All three run against an OFF-SCREEN-BUT-MAPPED root (see _offscreen below),
never a withdrawn one: a withdrawn CTk root leaves even a label with room to
spare at alloc=(1,1) (probed directly), which is the same ruling
render_pages.py's own module docstring documents for --audit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.render_pages import find_clipped_text_widgets  # noqa: E402


def _offscreen(root, width=400, height=200):
    """Map `root` off-screen: real geometry, never on the user's actual screen.

    Mirrors render_pages.py's own --audit driver exactly (same ruling, same
    fix): deiconify at a position far outside any real display, THEN let Tk
    process the pending geometry requests.
    """
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
