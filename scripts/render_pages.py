"""See every V6 page, and either photograph it or mechanically check it for clipped text.

    python scripts/render_pages.py <copy.db> <outdir>            # PNG capture (needs a real screen)
    python scripts/render_pages.py <copy.db> <outdir> --audit    # clipped-text widget audit, no capture
    python scripts/render_pages.py <copy.db>          --show     # open on screen, mainloop, no capture

PNG capture and --show use `build_views`: every page in `Sidebar.ITEMS`, plus the Model page loaded
for the model with the most cached process findings, with its Findings tab selected (Task 7's own
navigation: `app.set_model_route(model, tab="findings"); app.show_page("model")`) -- the Sidebar.ITEMS
pass alone only ever shows the Model page's EMPTY state (no route is set).

--audit uses its OWN, richer procedure (`run_audit`), extended in a pre-review fix round (2026-09-24)
after the first cut only ever saw whichever Model-page TAB happened to already be mapped (see point 2
below) -- and the facelift spec names the Model page, not any one tab of it, as the most likely place
for the bigger Step 1 text (SIZE_CAPTION 12, BODY 14, HEADING 17, TITLE 22, READOUT 20) to overflow an
unchanged layout:
  * every `Sidebar.ITEMS` page, plain;
  * the Model page for TWO independently-resolved models -- `resolve_findings_model` (most cached
    findings) and `resolve_ft_heavy_model` (most final-test rows linked to a trim: a deliberately
    different data shape, so the sweep is not one model's UI state twice) -- each with EVERY tab its
    ThemedTabView has registered selected in turn, by NAME, labeled "model:<tab>" / "model2:<tab>";
  * the Findings PAGE (not the Model page's Findings tab) with its first row opened, the densest
    layout on that page: a narrative, a settings table, and the "Open <model>" teal button -- opened
    by `_open_first_row`, which VERIFIES the row ends open rather than assuming a toggle worked (a
    review round, same date, found the naive version silently CLOSED an already-open row at the
    second window size and still reported success; see that function's own docstring);
  * optionally, one Model-page pass with a loader forced to fail (`_force_one_loader_failure`), so
    `_load_banner` -- otherwise never exercised, since the real database never fails a loader -- is
    checked against real rendered text instead of fixed by analogy to a sibling banner alone.

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
     says is not currently on screen. This is WHY the first cut of --audit only ever saw the Findings
     tab reliably (plus, incidentally, Drift Metrics and Trim vs Final Test -- probed directly with
     winfo_ismapped()/winfo_manager() down each tab's ancestor chain: both are CTkScrollableFrame
     subclasses whose content is embedded onto an internal canvas via create_window(), which does not
     un-map the same way a plain grid_forgotten child does). `_sweep_model_tabs` now selects every
     registered tab NAME itself (`page._tabs.set(name)`, CTkTabview's own public API -- the same call
     `ModelPage._select_tab` makes for the one name it knew) and audits right after, so each tab is
     measured in the state it is ACTUALLY in when selected, not left to however this quirk happened to
     leave it mapped. A tab genuinely never selected (impossible now, inside one sweep) would still
     correctly read alloc=(1,1) and be skipped -- proven in tests/test_render_pages_audit.py.

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
    """Run the Tk event loop for the full `seconds` so background loads land.

    Every page loader here runs on a worker thread and posts its result back
    through `app.ui` (UiDispatcher) rather than touching Tk directly (workers
    never call Tk -- see gui/v6/ui_dispatch.py); only `app.update()`, on the
    main thread, drains that queue.

    No early exit. This used to return as soon as the dispatcher's queue was
    empty (after a 1s floor) -- and an EMPTY queue does not mean the worker is
    DONE; it can just as well mean the worker has not reached its first
    `self.safe_after(apply)` yet (still doing synchronous DB work), which
    looks identical from here. Proven wrong empirically in the pre-review fix
    round (2026-09-24): switching the Model page straight from one model to
    another (6607, this database's highest-volume model -- see DENSE in
    chart_qa_render_all.py) took ~3.1s for its stats table to actually
    update; the old early exit walked the PREVIOUS model's still-displayed
    content at ~1s and mislabelled it as the new model's. A false negative
    -- silently auditing the wrong page state -- is worse than the extra
    wall-clock time always pumping the full budget costs.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update()
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


def resolve_ft_heavy_model(db, *, exclude: Optional[str] = None) -> Optional[str]:
    """The model with the most final-test rows LINKED to a trim record on this
    database -- a deliberately DIFFERENT data shape from resolve_findings_model
    (final-test match coverage, not findings volume), so the extended --audit
    sweep (pre-review fix round, 2026-09-24: "coverage is not one model's")
    exercises a second, independently-chosen page state rather than the same
    model's UI twice under a different label. Resolved by QUERY, same rule as
    resolve_findings_model and chart_qa_render_all.py's resolve_unit_fixtures:
    a hard-coded model name goes stale the day the database is rebuilt.

    `exclude`, when given, is left OUT of consideration first, falling back to
    including it only if it turns out to be the ONLY model with any linked
    final-test rows at all. This is what makes "coverage is not one model's" a
    property of the result, not just of the query: on the real database,
    resolve_findings_model and the un-excluded form of this query both pick
    '8232-1' (it is both the most-findings AND the most-FT-linked model, being
    the highest-volume customer-facing model this repo's data leans on
    throughout) -- picking a second model genuinely takes this into account
    rather than re-testing '8232-1' under a second label and calling it two
    data shapes.
    """
    from sqlalchemy import func
    from laser_trim_analyzer.database.models import (
        AnalysisResult as DBAR, FinalTestResult as DBFT)
    with db.session() as s:
        base = (s.query(DBAR.model, func.count(DBFT.id))
                .join(DBFT, DBFT.linked_trim_id == DBAR.id)
                .group_by(DBAR.model))
        if exclude is not None:
            row = (base.filter(DBAR.model != exclude)
                   .order_by(func.count(DBFT.id).desc(), DBAR.model).first())
            if row is not None:
                return row[0]
        row = base.order_by(func.count(DBFT.id).desc(), DBAR.model).first()
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


def _walk_page(app, page_key: str, label: str, size_label: str,
                clipped: List[ClippedWidget]) -> None:
    page = app.page_container.get_page(page_key)
    if page is not None:
        clipped.extend(find_clipped_text_widgets(page, page=label, window_size=size_label))


def _sweep_model_tabs(app, model: str, label_prefix: str, size_label: str,
                       clipped: List[ClippedWidget]) -> None:
    """Show the Model page for `model`, then select EVERY tab its ThemedTabView has
    registered, BY NAME (never hard-coded -- the registered order/set can change),
    pumping and auditing each in turn, labeled "<label_prefix>:<tab name>".

    Pre-review fix round (2026-09-24): the original --audit only ever saw whichever
    tab happened to already be mapped (Findings, plus incidentally Drift Metrics and
    Trim vs Final Test -- see the module docstring). CTkTabview grids only the ACTIVE
    tab (`.set()` grids the new one and grid_forgets the rest 100ms later), so a tab
    never selected is a tab never measured -- and the spec names the Model page, not
    any one tab of it, as the most likely overflow. `page._tabs.set(name)` is
    CTkTabview's own public API (the same call `ModelPage._select_tab` makes for the
    one name it knows, "Findings"); calling it directly for every registered name
    exercises the identical mechanism for all seven.

    Always detours through Home first: `PageContainer.show()` no-ops when the
    requested page is already current, so calling this twice in a row (a second
    model, right after the first model's sweep leaves "model" current) would
    otherwise leave `on_show()` never re-fired and the new model's route unconsumed.
    """
    app.show_page("home")
    app.set_model_route(model)
    app.show_page("model")
    page = app.page_container.get_page("model")
    _pump(app)
    app.update_idletasks()
    for name in list(page._tabs._name_list):
        page._tabs.set(name)
        _pump(app)
        app.update_idletasks()
        clipped.extend(find_clipped_text_widgets(
            page, page=f"{label_prefix}:{name}", window_size=size_label))


def _open_first_row(view) -> bool:
    """Open `view`'s first row, verifying it actually ends open; never a blind
    toggle. Pure -- takes a FindingsView directly (not the app), so this is the
    part tests/test_render_pages_audit.py drives on its own, no page/app needed.

    Review finding (2026-09-24): FindingsView._render() re-opens whatever row
    was open before a refresh (`was_open` -> `toggle(was_open)`), and the
    Findings PAGE gets shown more than once in one --audit run -- once per
    window size's Sidebar.ITEMS pass, once again for this explicit open. By
    the SECOND window size, the row this function opened for the FIRST size is
    routinely already open again by the time this runs (re-opened by
    _render()'s own "keep it open across a refresh" behaviour) -- and
    `toggle()` on an ALREADY-open key CLOSES it. The old code called
    `view.toggle(key)` unconditionally and returned True regardless, so at
    1280x720 -- the size every clip in this task was found at -- the densest
    layout on the page (the settings table, the teal "Open <model>" button)
    was silently never checked at all while "0 clipped" read as clean.

    Fixed two ways, not one: (1) toggle only when the row is not ALREADY open
    (so a second call in the same state is a no-op, not a close); (2) the
    return value is VERIFIED against the view's actual state afterward
    (`open_key == key and a detail pane exists`), never assumed from having
    called toggle. Either fix alone would have been enough for the reproduced
    bug; both together also cover a toggle that silently fails for some other
    reason (e.g. `row_widgets` losing the key between the check and the call).
    """
    if not view.row_widgets:
        return False
    key = next(iter(view.row_widgets))
    if view.open_key != key:
        view.toggle(key)
    return view.open_key == key and view._detail is not None


def run_audit(app, target_model: Optional[str],
               ft_model: Optional[str]) -> Tuple[List[ClippedWidget], List[str]]:
    """Walk every view at every audited size; return (every clip found, every
    audit-tooling FAILURE -- a state this script could not itself get the app
    into, as distinct from a clip, which is the app's own text being cut off).

    `target_model` (most cached findings) and `ft_model` (most final-test rows
    linked to a trim -- deliberately a different data shape) each get the FULL
    Model-page tab sweep, labeled "model:<tab>" and "model2:<tab>" respectively,
    so the densest page in the app is checked against two independently-chosen
    real data shapes, not one.
    """
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar

    clipped: List[ClippedWidget] = []
    failures: List[str] = []
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
        for key, _label in Sidebar.ITEMS:
            app.show_page(key)
            _pump(app)
            app.update_idletasks()
            _walk_page(app, key, key, size_label, clipped)
        if target_model:
            _sweep_model_tabs(app, target_model, "model", size_label, clipped)
        if ft_model:
            _sweep_model_tabs(app, ft_model, "model2", size_label, clipped)
        # Findings PAGE with its first row opened -- the current page is "model"
        # (or "home", if neither model resolved), never "findings", so this is
        # always a real transition; no detour needed.
        app.show_page("findings")
        _pump(app)
        page = app.page_container.get_page("findings")
        view = getattr(page, "_view", None)
        if view is None or not view.row_widgets:
            print(f"note: the Findings page has no rows to open at {size_label} -- "
                  "skipping the opened-row audit for this size")
        elif not _open_first_row(view):
            # NEVER a silent skip (review finding): rows existed and opening one
            # still did not work, which means the densest layout on this page
            # (settings table, teal button) went unchecked at this size -- that
            # is a failure of THIS SCRIPT, reported the same way a real clip is
            # (a line in audit.txt, counted toward a non-zero exit), not folded
            # into "0 clipped" where it would read as clean.
            msg = (f"AUDIT FAILURE | {size_label} | could not open the Findings "
                   f"page's first row (view.open_key={view.open_key!r}) -- the "
                   f"opened-detail state was never actually checked at this size")
            print(msg)
            failures.append(msg)
        else:
            _pump(app)
            app.update_idletasks()
            _walk_page(app, "findings", "findings:opened", size_label, clipped)
    # Optional (review, 2026-09-24): _load_banner ("Could not load: ...") was
    # fixed in the base report by ANALOGY to _spec_banner two lines above it in
    # the same file -- the real database never fails a loader, so it had never
    # actually been rendered with real text. Force exactly one to fail here so
    # it is proven, not just reasoned about. Once, at the smaller audited size
    # only (every clip in this task was found there) -- a supplementary check,
    # not part of the required sweep, so it does not double the cost of the
    # whole run.
    if target_model:
        width, height = _SMALL_SIZE
        size_label = f"{width}x{height}"
        app.geometry(f"{width}x{height}+20000+20000")
        app.deiconify()
        app.update_idletasks()
        app.update()
        _force_one_loader_failure(app, target_model, size_label, clipped)
    return clipped, failures


def _force_one_loader_failure(app, model: str, size_label: str,
                               clipped: List[ClippedWidget]) -> None:
    """Patch ModelPage._load_units to always raise for the duration of ONE
    reload, so `failed` (the plain list `_set_load_banner` reads) is genuinely
    non-empty and the banner renders real text -- restored in a `finally` no
    matter what, so the patch can never leak into any other page or model this
    script still has to audit. `_load_units` feeds "unit list" into `failed`
    (gui/v6/pages/model_page.py:371) and is only otherwise called from a
    search-box handler this audit never triggers (:907), so patching it here
    does not disturb anything else this run measures.
    """
    from laser_trim_analyzer.gui.v6.pages.model_page import ModelPage

    def _always_fails(self, model):
        raise RuntimeError("render_pages.py --audit: forced failure to exercise _load_banner")

    original = ModelPage._load_units
    ModelPage._load_units = _always_fails
    try:
        app.show_page("home")
        app.set_model_route(model)
        app.show_page("model")
        _pump(app)
        app.update_idletasks()
        page = app.page_container.get_page("model")
        clipped.extend(find_clipped_text_widgets(
            page, page="model:load-banner-forced", window_size=size_label))
    finally:
        ModelPage._load_units = original


def _run_audit_mode(db_path: Path, outdir: Path) -> int:
    app, db = _build_app(db_path)
    try:
        target_model = resolve_findings_model(db)
        ft_model = resolve_ft_heavy_model(db, exclude=target_model)
        if target_model is None:
            print("note: this database has no findings and no analysis rows -- "
                  "auditing every Sidebar.ITEMS page, but not the Model page's loaded state")
        if ft_model is None:
            print("note: this database has no final-test rows linked to a trim -- "
                  "skipping the second model's tab sweep")
        elif ft_model == target_model:
            print(f"note: {target_model!r} is the ONLY model on this database with any "
                  f"final-test rows linked to a trim, so it is also the most-linked-FT-"
                  f"rows model even excluding itself -- the second sweep (model2:<tab>) "
                  f"re-tests it, which is still real coverage, just not a second model")
        app.withdraw()      # never flash on-screen before run_audit positions it off-screen
        n_sizes = len(_audit_sizes(app))
        clipped, failures = run_audit(app, target_model, ft_model)
    finally:
        app.destroy()
        db.close()

    outdir.mkdir(parents=True, exist_ok=True)
    # Failures first: a state this script could not verify at all outranks a
    # confirmed clip -- and either one means the run is not clean, so both
    # count toward the exit code together (never "0 clipped" alone).
    lines = list(failures) + [c.line() for c in clipped]
    header = (f"{len(clipped)} clipped widget(s), {len(failures)} audit failure(s), "
              f"across {n_sizes} window size(s); "
              f"model (most findings): {target_model!r}; "
              f"model2 (most linked final-test rows): {ft_model!r}")
    (outdir / "audit.txt").write_text(header + "\n" + "\n".join(lines) + ("\n" if lines else ""))
    print(header)
    for line in lines:
        print(line)
    print(f"-> {outdir / 'audit.txt'}")
    return 1 if (clipped or failures) else 0


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
