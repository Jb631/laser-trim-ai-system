"""Chart text and colours come from the theme, like everything else (spec section 1)."""
import pathlib
import re

V6 = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"
QA_RENDER_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/chart_qa_render_all.py"


def test_no_chart_font_size_is_a_literal():
    offenders = []
    for p in (V6 / "widgets").glob("*.py"):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"\b(fontsize|labelsize)\s*=\s*[0-9]", line):
                offenders.append(f"{p.name}:{n}: {line.strip()}")
    assert not offenders, offenders


def test_no_hex_colour_outside_the_theme():
    offenders = []
    for p in V6.rglob("*.py"):
        if p.name == "theme.py":
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r'"#[0-9a-fA-F]{6}"', line):
                offenders.append(f"{p.relative_to(V6)}:{n}")
    assert not offenders, offenders


def test_the_table_column_widths_were_left_alone():
    """Grid column widths, not fonts: the font-token task left them 190 and 78. Since the final
    review (2026-09-24) they are scaled like every other size (grid minsize is real pixels to Tk,
    so at 150% Windows scaling an unscaled one stays 190 px while the text grows) -- the same
    two widths, in CustomTkinter's unscaled units."""
    src = (V6 / "widgets" / "stats_table.py").read_text()
    assert "minsize=frame._apply_widget_scaling(190)" in src
    assert "minsize=frame._apply_widget_scaling(78)" in src


def test_the_qa_render_script_has_no_copied_palette():
    """The headless render harness used to hand-copy the palette into its own
    `_Theme` stub, which went stale the moment theme.py's colours changed
    (review finding, 2026-09-23) -- every Step-5 render then validated
    colours the app no longer draws. Read as TEXT, never imported: the
    script installs fake tkinter/customtkinter modules into sys.modules at
    import time, which would poison every other test in this process."""
    src = QA_RENDER_SCRIPT.read_text()
    offenders = [f"line {n}: {line.strip()}"
                 for n, line in enumerate(src.splitlines(), 1)
                 if re.search(r'"#[0-9a-fA-F]{6}"', line)]
    assert not offenders, offenders
    assert "ThemeManager" in src


def test_the_qa_render_script_loads_the_bundled_fonts():
    """Facelift step 2 Task 3b (2026-09-24): the harness never called
    font_loader.load_bundled_fonts(), so its PNGs were DejaVu Sans -- not
    what the app draws once the bundled fonts load. Read as text (see the
    note above): the call must sit at true module scope, before any chart
    gets built, not tucked inside `if __name__ == "__main__":` where
    importing this module for some other purpose would never trigger it."""
    src = QA_RENDER_SCRIPT.read_text()
    assert "from laser_trim_analyzer.gui.v6.font_loader import load_bundled_fonts" in src
    call_at = src.index("load_bundled_fonts()")
    guard_at = src.index('if __name__ == "__main__":')
    first_chart_factory_at = src.index("def _focus(")
    assert call_at < guard_at, "must run at import time, not only under __main__"
    assert call_at < first_chart_factory_at, "must load before any chart gets built"
