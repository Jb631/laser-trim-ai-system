"""The finish pass on the pages (option B, 2026-10-04): names, dates, numbers, buttons, the Model
page's top, the charts and the Process page.

James, with Task Manager TMOG open: "now this is an example of a finished peice of solftware". The
screenshots taken that day (docs/superpowers/specs/2026-10-04-option-b-design.md, "The finish list")
found three date formats, a page called two names, 0.005201 beside 0.011, flat text dressed as
buttons, bright blue dropdown arrows, four lines of red and grey before a chart, a fail-rate axis to
125% and a run-on line of network paths. Each test here pins one of those gone. Every example value
is invented.
"""
import ast
import pathlib
import re
from datetime import datetime

import pytest

V6 = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"


def _walk(widget):
    """Every widget Tk has under `widget` -- tkinter's own winfo_children (CustomTkinter's hides a
    tab view's tabs and a widget's internal parts)."""
    import tkinter
    yield widget
    for child in tkinter.Misc.winfo_children(widget):
        yield from _walk(child)


def _texts(widget):
    out = []
    for w in _walk(widget):
        try:
            text = w.cget("text")
        except Exception:
            continue
        if isinstance(text, str) and text:
            out.append(text)
    return out


def _shown_literals(path: pathlib.Path):
    """Every string literal in `path` a person could be SHOWN: not a docstring, and not the message
    of a logging call (a log line is never on screen -- "Dashboard: yield query failed" stays a
    log key)."""
    tree = ast.parse(path.read_text())
    hidden = {id(node.value) for node in ast.walk(tree)
              if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("debug", "info", "warning", "error", "exception")):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant):
                    hidden.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in hidden:
            yield node.lineno, node.value


# ---- 1. "Company trends" everywhere (the page the Overview's link opens) ---------------------

def test_the_company_trends_page_is_called_company_trends(make_app):
    """The Overview's link says "Company trends" and the page it opened was titled "Dashboard". The
    key stays "dashboard" (every deep link navigates by key); what a person reads changes."""
    app = make_app()
    page = app.page_container.get_page("dashboard")
    assert page.page_title == "Company trends"
    app.show_page("dashboard")
    page.reload_now()
    assert not [t for t in _texts(page) if "Dashboard" in t]


def test_no_screen_in_the_app_says_dashboard():
    """Every string a V6 screen can show -- the stats table's note about where unit yield lives, the
    database clean-up's "open it again" line -- names the page "Company trends".

    app.py is left out: it registers the page under its key and passes it the key's old word as a
    title, which the page no longer reads (it names itself -- the test above)."""
    offenders = [f"{p.relative_to(V6)}:{line}: {text[:60]!r}"
                 for p in V6.rglob("*.py") if p.name != "app.py"
                 for line, text in _shown_literals(p)
                 if "Dashboard" in text]
    assert not offenders, offenders


def test_the_scan_for_shown_text_skips_log_lines_and_docstrings_only(tmp_path):
    src = tmp_path / "x.py"
    src.write_text('"""Dashboard docstring."""\n'
                   'logger.exception("Dashboard: yield query failed")\n'
                   'label = "see the Dashboard"\n')
    assert [text for _line, text in _shown_literals(src)] == ["see the Dashboard"]


# ---- 2. "Settings → Backlog", never the Pricing card that merged into it (2026-09-20) ---------

def test_the_money_table_sends_you_to_settings_backlog(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.priorities_panel import PrioritiesPanel
    panel = PrioritiesPanel(tk_root, theme=ThemeManager(), on_row_click=lambda m: None)
    panel.set_rows([
        {"model": "INV-1", "ft_total": 40, "ft_fails": 4, "ft_fail_rate": 10.0,
         "price": 12.5, "dollar_impact": 25.0},
        {"model": "INV-2", "ft_total": 30, "ft_fails": 3, "ft_fail_rate": 10.0,
         "price": None, "dollar_impact": None}])
    said = " ".join(_texts(panel))
    assert "Settings → Backlog" in said
    assert "Pricing" not in said


def test_the_cost_priorities_docstring_names_the_backlog_card():
    from laser_trim_analyzer.core import cost_priorities
    doc = cost_priorities.__doc__
    assert not re.search(r"Settings\s*→\s*Pricing", doc)
    assert "Settings → Backlog" in doc
