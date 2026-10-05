"""Make the app's icon images from the Tabler icon font -- run once, and again only to add an icon.

    python scripts/build_icons.py <folder with tabler-icons.ttf and tabler-icons.css>

The Tabler icons (MIT licence, https://tabler.io/icons) come as a font; the app needs small images,
which CustomTkinter shows on any platform with no font to install. Downloaded 2026-10-04 (James:
"you can download") from https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/fonts/
tabler-icons.ttf and .../dist/tabler-icons.css; only the images below are kept, white on clear at
64 px -- gui/v6/icons.py tints and sizes them. The font itself is not shipped.
"""
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "src" / "laser_trim_analyzer" / "gui" / "v6" / "icons"
SIZE = 64

# The app's name for an icon -> Tabler's.
ICONS = {
    "overview": "layout-dashboard", "models": "list-details", "settings": "settings",
    "process": "file-import", "trends": "chart-line", "findings": "bulb", "database": "database",
    "ok": "circle-check", "warning": "alert-triangle", "clock": "clock", "refresh": "refresh",
    "open": "arrow-up-right", "dollar": "currency-dollar", "drift": "activity", "copy": "copy",
    "export": "file-spreadsheet", "folder": "folder", "play": "player-play", "stop": "player-stop",
    "search": "search", "chevron_down": "chevron-down", "chevron_right": "chevron-right",
}


def main(src: Path) -> int:
    css = (src / "tabler-icons.css").read_text(encoding="utf-8")
    codes = dict(re.findall(r'\.ti-([a-z0-9-]+):before\s*\{\s*content:\s*"\\([0-9a-f]+)"', css))
    font = ImageFont.truetype(str(src / "tabler-icons.ttf"), size=int(SIZE * 0.86))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, tabler in ICONS.items():
        char = chr(int(codes[tabler], 16))
        img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
        draw.text(((SIZE - (right - left)) / 2 - left, (SIZE - (bottom - top)) / 2 - top), char,
                  font=font, fill=(255, 255, 255, 255))
        img.save(OUT / f"{name}.png")
    print(f"{len(ICONS)} icons -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
