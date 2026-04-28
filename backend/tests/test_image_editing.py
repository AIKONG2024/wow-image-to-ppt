from PIL import Image, ImageDraw

from app.image_editing import erase_regions
from app.models import BBox


def test_erase_regions_keeps_art_between_large_separate_text_boxes():
    image = Image.new("RGBA", (260, 140), "#ffd21c")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 45, 70, 95], fill="black")
    boxes = [
        BBox(x=110, y=16, width=120, height=28),
        BBox(x=110, y=56, width=120, height=28),
        BBox(x=110, y=96, width=120, height=28),
    ]

    erased = erase_regions(image, BBox(x=0, y=0, width=260, height=140), boxes)

    assert erased.getpixel((45, 70))[:3] == (0, 0, 0)
    assert erased.getpixel((150, 70))[:3] == (255, 210, 28)


def test_erase_regions_fills_large_top_text_tail_without_smearing():
    image = Image.new("RGBA", (260, 220), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rectangle([4, 4, 98, 64], fill="black")
    draw.rectangle([150, 140, 210, 190], fill="#d71920")

    erased = erase_regions(
        image,
        BBox(x=0, y=0, width=260, height=220),
        [BBox(x=0, y=0, width=112, height=76)],
    )

    assert erased.getpixel((40, 30))[:3] == (248, 250, 252)
    assert erased.getpixel((180, 165))[:3] == (215, 25, 32)
