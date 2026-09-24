"""Chart text and colours come from the theme, like everything else (spec section 1)."""
import pathlib
import re

V6 = pathlib.Path(__file__).resolve().parents[1] / "src/laser_trim_analyzer/gui/v6"


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
    src = (V6 / "widgets" / "stats_table.py").read_text()
    assert "minsize=190" in src and "minsize=78" in src
