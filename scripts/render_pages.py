"""See every V6 page, and either photograph it or mechanically check it for clipped text.

    python scripts/render_pages.py <copy.db> <outdir>            # PNG capture (needs a real screen)
    python scripts/render_pages.py <copy.db> <outdir> --audit    # clipped-text widget audit, no capture
    python scripts/render_pages.py <copy.db>          --show     # open on screen, mainloop, no capture

Every mode shows the same set of views: every page in `Sidebar.ITEMS`, plus the Model page loaded
for the model with the most cached process findings, with its Findings tab selected (Task 7's own
navigation: `app.set_model_route(model, tab="findings"); app.show_page("model")`) -- the Sidebar.ITEMS
pass alone only ever shows the Model page's EMPTY state (no route is set), so this second pass is the
one that actually exercises the page the facelift spec names as the most likely place for the bigger
Step 1 text (SIZE_CAPTION 12, BODY 14, HEADING 17, TITLE 22, READOUT 20) to overflow an unchanged layout.

Why two modes exist (controller ruling, Task 9, 2026-09-24): PIL.ImageGrab.grab is REFUSED on this
Mac -- confirmed empirically, see _grab() -- because the process has no Screen Recording permission,
which is James's security setting to grant, not this script's to request. The default (capture) mode
stays for a machine that allows it (Windows, at work) and fails loudly rather than saving a blank PNG
when it can't. --audit is the mechanical replacement for "look at the PNG and check nothing is cut
off": it walks the real widget tree and reports every text widget Tk is squeezing smaller than the
text actually needs.

The --audit trick, in one paragraph: a widget's ALLOCATED size (`winfo_width`/`winfo_height`) is what
its container actually gave it; its REQUESTED size (`winfo_reqwidth`/`winfo_reqheight`) is what it
would need to draw its current text without clipping. When allocated is smaller than requested, the
text is being cut off -- that comparison alone is the whole detector (find_clipped_text_widgets,
proven against real CustomTkinter geometry in tests/test_render_pages_audit.py). Two things have to be
true for that comparison to mean anything, both confirmed empirically before this was written:
  1. The window must be MAPPED, even if invisible. A withdrawn root leaves every child stuck at
     Tk's placeholder size (1x1) forever, fit or not -- probed directly: a label with plenty of room
     read alloc=(1,1) under a withdrawn root. So --audit deiconifies the window OFF-SCREEN
     ("+20000+20000", far past any real display) instead of withdrawing it: mapped, just not in
     anyone's way.
  2. Only CURRENTLY MAPPED widgets count. CTkTabview only grids its ACTIVE tab (`.set()` grids the
     new one and grid_forgets the rest 100 ms later); a tab nobody has ever selected was never gridded
     at all and sits at that same (1x1) placeholder no matter how much text it holds. Reporting that
     as a clip would be pure noise, so find_clipped_text_widgets skips anything `winfo_ismapped()`
     says is not currently on screen -- which also means --audit only ever sees the Findings tab of
     the Model page, not its other six tabs; those are unchanged from before this facelift and were
     not in this task's scope.

Mirrors scripts/refresh_findings.py for the database guard and the double global injection.
"""
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import _db_guard

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# "more than 1 px" (controller ruling) -- so DPI/scaling rounding never counts as a clip.
_TOLERANCE_PX = 1

# The second size the ruling names, checked in addition to the configured window
# (config.gui.window_width x window_height, 1400x900 by default -- see _audit_sizes).
_SMALL_SIZE = (1280, 720)

# How long to pump the event loop per view so a page's background loaders land
# (ModelPage._reload, the selector refresh, etc. all run on worker threads and
# post their results back through UiDispatcher -- see gui/v6/ui_dispatch.py).
_PUMP_SECONDS = 5.0


# ---------------------------------------------------------------------------
# The detector. Pure tkinter -- no customtkinter or app import required to USE
# it, so it is unit-testable on its own (tests/test_render_pages_audit.py).
# ---------------------------------------------------------------------------

@dataclass
class ClippedWidget:
    page: str
    window_size: str
    path: str
    text: str
    alloc: Tuple[int, int]
    req: Tuple[int, int]

    def line(self) -> str:
        aw, ah = self.alloc
        rw, rh = self.req
        return (f"{self.page} | {self.window_size} | {self.path} | "
                f"{self.text!r} | alloc={aw}x{ah} req={rw}x{rh}")


def _iter_widgets(root) -> Iterator[tk.Misc]:
    """Depth-first walk of `root` and every Tk descendant (real widgets only --
    this is `winfo_children()`, so it reaches CTk composites' internal tk
    widgets exactly as Tk itself sees them)."""
    yield root
    try:
        children = root.winfo_children()
    except Exception:
        children = []
    for child in children:
        yield from _iter_widgets(child)


def find_clipped_text_widgets(root, *, page: str = "", window_size: str = "") -> List[ClippedWidget]:
    """Every text-bearing widget under `root` whose allocated width or height is
    more than `_TOLERANCE_PX` below its requested (natural) size -- i.e. its
    text is being cut off by its container.

    Only `tkinter.Label` and `tkinter.Button` are checked. CustomTkinter draws
    a widget's text with a REAL tk widget of one of those two classes, held
    inside its own frame/canvas shell: `CTkLabel._label`, `CTkButton.
    _text_label`, `CTkOptionMenu._text_label` -- and each CTkSegmentedButton
    segment is itself a CTkButton with its own `_text_label`. So walking every
    descendant and checking these two plain classes reaches every label this
    app draws, composite or not, with no per-CTk-class special case. Checked
    empirically against the pinned customtkinter (5.2.2): every one of those
    inner widgets is in fact a `tkinter.Label`; `tkinter.Button` is kept in the
    check anyway (cheap, and future-proof against a widget that really is one).

    Skips anything `winfo_ismapped()` says is not currently on screen -- see
    the module docstring's point 2. Not a clip if it is not visible.
    """
    found: List[ClippedWidget] = []
    for widget in _iter_widgets(root):
        if not isinstance(widget, (tk.Label, tk.Button)):
            continue
        try:
            raw_text = widget.cget("text")
        except Exception:
            continue
        text = "" if raw_text is None else str(raw_text)
        if not text:
            continue
        try:
            if not widget.winfo_ismapped():
                continue
            alloc_w, alloc_h = widget.winfo_width(), widget.winfo_height()
            req_w, req_h = widget.winfo_reqwidth(), widget.winfo_reqheight()
        except Exception:
            continue
        if (req_w - alloc_w > _TOLERANCE_PX) or (req_h - alloc_h > _TOLERANCE_PX):
            found.append(ClippedWidget(
                page=page, window_size=window_size, path=str(widget),
                text=text[:60], alloc=(alloc_w, alloc_h), req=(req_w, req_h),
            ))
    return found


# ---------------------------------------------------------------------------
# Driving the real app.
# ---------------------------------------------------------------------------

def _pump(app, seconds: float = _PUMP_SECONDS) -> None:
    """Run the Tk event loop for up to `seconds` so background loads land.

    Every page loader here runs on a worker thread and posts its result back
    through `app.ui` (UiDispatcher) rather than touching Tk directly (workers
    never call Tk -- see gui/v6/ui_dispatch.py); only `app.update()`, on the
    main thread, drains that queue. Exits early, after a 1s floor, once the
    dispatcher's own queue is empty -- most pages settle well under 5s and
    there are up to 16 of these (8 views x 2 window sizes) in one run.
    """
    deadline = time.monotonic() + seconds
    floor = time.monotonic() + min(1.0, seconds)
    while True:
        app.update()
        now = time.monotonic()
        if now >= deadline:
            return
        if now >= floor:
            try:
                if app.ui._q.empty():
                    return
            except Exception:
                pass
        time.sleep(0.02)


def _build_app(db_path: Path):
    """Refuse the production database (caller's job, before this is called),
    then build a V6App against `db_path` with BOTH database globals injected --
    mirrors scripts/refresh_findings.py, and the same global constraint every
    script/test that builds a DatabaseManager or Processor follows."""
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    from laser_trim_analyzer.config import Config
    from laser_trim_analyzer.gui.v6.app import V6App

    db = mgr.DatabaseManager(db_path)
    mgr._db_manager = db
    dbpkg._db_manager = db
    config = Config()
    config.database.path = db_path
    app = V6App(config, db=db, auto_train_on_first_run=False)
    return app, db


def resolve_findings_model(db) -> Optional[str]:
    """The model with the most cached process findings (tie-break: alphabetical).

    Falls back to the model with the most analysis rows when the copied
    database carries no findings cache. Resolved by QUERY, never a literal
    model name -- a hard-coded one would silently stop meaning anything the
    day the work database is rebuilt (the same trap chart_qa_render_all.py's
    fixtures hit before they were switched to resolve_unit_fixtures()).
    """
    from collections import Counter

    findings = db.get_process_findings()
    if findings:
        counts = Counter(f["model"] for f in findings if f.get("model"))
        if counts:
            best = max(counts.values())
            return min(m for m, c in counts.items() if c == best)

    from sqlalchemy import func
    from laser_trim_analyzer.database.models import AnalysisResult as DBAR
    with db.session() as s:
        row = (s.query(DBAR.model, func.count(DBAR.id))
               .group_by(DBAR.model)
               .order_by(func.count(DBAR.id).desc(), DBAR.model)
               .first())
    return row[0] if row else None


# Each view: (report label, PNG filename stub, page_container key, setup(app)).
View = Tuple[str, str, str, Callable]


def build_views(target_model: Optional[str]) -> List[View]:
    """Every Sidebar.ITEMS page, plus the Model page for `target_model` with its
    Findings tab selected -- see the module docstring for why the second one
    is a separate view rather than folded into the Sidebar.ITEMS pass.

    Order matters: `PageContainer.show()` no-ops when the requested page is
    already current, so the extra Model-page view MUST come after some other
    page has been shown (it does -- it is appended after the whole
    Sidebar.ITEMS loop, and "process", the last entry, is never "model").
    """
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar

    views: List[View] = [
        (key, key, key, (lambda app, key=key: app.show_page(key)))
        for key, _label in Sidebar.ITEMS
    ]
    if target_model:
        def _model_findings(app, model=target_model):
            app.set_model_route(model, tab="findings")
            app.show_page("model")
        views.append((f"model ({target_model}) — Findings tab", "model-findings",
                      "model", _model_findings))
    return views


def _audit_sizes(app) -> List[Tuple[int, int]]:
    return [(app.config.gui.window_width, app.config.gui.window_height), _SMALL_SIZE]


def run_audit(app, target_model: Optional[str]) -> List[ClippedWidget]:
    """Walk every view at every audited size; return every clip found."""
    clipped: List[ClippedWidget] = []
    views = build_views(target_model)
    for width, height in _audit_sizes(app):
        size_label = f"{width}x{height}"
        # Off-screen but MAPPED -- see the module docstring, point 1.
        app.geometry(f"{width}x{height}+20000+20000")
        app.deiconify()
        app.update_idletasks()
        app.update()
        # The sidebar is not inside any page's subtree, and it never changes
        # across a page switch, so it gets one walk per size rather than one
        # per view.
        clipped.extend(find_clipped_text_widgets(app.sidebar, page="sidebar", window_size=size_label))
        for report_label, _stub, page_key, setup in views:
            setup(app)
            _pump(app)
            app.update_idletasks()
            page = app.page_container.get_page(page_key)
            if page is not None:
                clipped.extend(find_clipped_text_widgets(page, page=report_label, window_size=size_label))
    return clipped


def _run_audit_mode(db_path: Path, outdir: Path) -> int:
    app, db = _build_app(db_path)
    try:
        target_model = resolve_findings_model(db)
        if target_model is None:
            print("note: this database has no findings and no analysis rows -- "
                  "auditing every Sidebar.ITEMS page, but not the Model page's loaded state")
        app.withdraw()      # never flash on-screen before run_audit positions it off-screen
        n_sizes = len(_audit_sizes(app))
        clipped = run_audit(app, target_model)
    finally:
        app.destroy()
        db.close()

    outdir.mkdir(parents=True, exist_ok=True)
    lines = [c.line() for c in clipped]
    header = (f"{len(clipped)} clipped widget(s) across {n_sizes} window size(s); "
              f"model under test: {target_model!r}")
    (outdir / "audit.txt").write_text(header + "\n" + "\n".join(lines) + ("\n" if lines else ""))
    print(header)
    for line in lines:
        print(line)
    print(f"-> {outdir / 'audit.txt'}")
    return 1 if clipped else 0


# ---------------------------------------------------------------------------
# PNG capture (brief's original interface -- needs a real, permitted screen).
# ---------------------------------------------------------------------------

def _grab(bbox):
    """PIL.ImageGrab.grab(bbox), or None on any failure (including macOS's
    Screen Recording refusal -- see the module docstring)."""
    try:
        from PIL import ImageGrab
        return ImageGrab.grab(bbox=bbox)
    except Exception:
        return None


def _is_blank(img) -> bool:
    """True when every pixel is the same colour -- a capture that 'succeeded'
    but drew nothing proves nothing, so it is treated the same as a refusal."""
    try:
        extrema = img.getextrema()
    except Exception:
        return True
    bands = extrema if isinstance(extrema, list) else [extrema]
    return all(lo == hi for lo, hi in bands)


def _safe_stub(stub: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in stub)


_REFUSAL = (
    "REFUSED: PIL.ImageGrab could not capture the window (blank image or a capture\n"
    "error) -- on macOS this means the process has no Screen Recording permission.\n"
    "Granting it is a security setting only the machine's owner can change; this\n"
    "script will not ask for it and will not save a blank PNG in its place.\n"
    "Use --audit instead (a mechanical clipped-text check, no capture needed), or\n"
    "run this capture mode on a machine that allows it (e.g. Windows, at work)."
)


def _run_capture_mode(db_path: Path, outdir: Path) -> int:
    app, db = _build_app(db_path)
    try:
        target_model = resolve_findings_model(db)
        outdir.mkdir(parents=True, exist_ok=True)
        app.deiconify()
        app.update_idletasks()
        app.update()
        saved = []
        for report_label, stub, _page_key, setup in build_views(target_model):
            setup(app)
            _pump(app)
            app.update_idletasks()
            app.update()
            x, y = app.winfo_rootx(), app.winfo_rooty()
            w, h = app.winfo_width(), app.winfo_height()
            img = _grab((x, y, x + w, y + h))
            if img is None or _is_blank(img):
                print(_REFUSAL)
                return 1
            path = outdir / f"{_safe_stub(stub)}.png"
            img.save(path)
            saved.append(path)
            print(f"saved {path.name}  ({report_label})")
        print(f"\n{len(saved)} PNGs -> {outdir}")
        return 0
    finally:
        app.destroy()
        db.close()


# ---------------------------------------------------------------------------
# --show: on screen, for a person to look at.
# ---------------------------------------------------------------------------

def _run_show_mode(db_path: Path) -> int:
    app, db = _build_app(db_path)
    app.deiconify()
    try:
        app.run()               # blocks in mainloop() until the window is closed
    finally:
        db.close()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)

    mode = "capture"
    if "--audit" in argv and "--show" in argv:
        print("usage: --audit and --show are mutually exclusive")
        return 2
    if "--audit" in argv:
        mode = "audit"
        argv = [a for a in argv if a != "--audit"]
    elif "--show" in argv:
        mode = "show"
        argv = [a for a in argv if a != "--show"]

    if not argv:
        print(__doc__)
        return 2
    db_path = Path(argv[0])

    if _db_guard.is_production_db(db_path, REPO, by_name=True):
        print(f"FATAL | {db_path} is the PRODUCTION database and this script opens it "
              f"read-write.\n"
              f"      | make a copy and pass that instead:\n"
              f"      |     cp data/analysis.db /tmp/qa_copy.db\n"
              f"      |     python scripts/render_pages.py /tmp/qa_copy.db qa_output/pages --audit")
        return 2
    if not db_path.exists():
        print(f"no such database: {db_path}")
        return 2

    if mode == "show":
        return _run_show_mode(db_path)

    if len(argv) < 2:
        print("usage: python scripts/render_pages.py <copy.db> <outdir> [--audit]\n"
              "(outdir is required for PNG capture and for --audit)")
        return 2
    outdir = Path(argv[1])

    if mode == "audit":
        return _run_audit_mode(db_path, outdir)
    return _run_capture_mode(db_path, outdir)


if __name__ == "__main__":
    raise SystemExit(main())
