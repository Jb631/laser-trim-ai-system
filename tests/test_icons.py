"""The icons: images made once from the Tabler set (MIT), tinted and sized for CustomTkinter."""
from PIL import Image

from laser_trim_analyzer.gui.v6 import icons


def test_every_icon_is_a_square_white_image_with_its_licence_beside_it():
    files = sorted(icons.ICON_DIR.glob("*.png"))
    assert len(files) >= 20
    assert (icons.ICON_DIR / "LICENSE-tabler.txt").is_file()
    for path in files:
        img = Image.open(path).convert("RGBA")
        assert img.size[0] == img.size[1] >= 48, path.name
        assert img.getchannel("A").getbbox() is not None, f"{path.name} is empty"


def test_an_icon_comes_in_the_colour_and_size_asked_for(tk_root):
    img = icons.icon("overview", "#3b82f6", 18)
    assert img.cget("size") == (18, 18)
    pil = img.cget("dark_image")
    opaque = [px for px in pil.getdata() if px[3] == 255]
    assert opaque and all(px[:3] == (0x3b, 0x82, 0xf6) for px in opaque)
    assert icons.icon("overview", "#3b82f6", 18) is img                 # one, shared


def test_a_missing_icon_is_text_only_never_a_crash(tk_root, caplog):
    assert icons.icon("no-such-icon", "#ffffff", 18) is None
    assert any("no-such-icon" in r.getMessage() for r in caplog.records)
