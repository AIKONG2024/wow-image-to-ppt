from PIL import Image, ImageDraw

from app.models import BBox, Component, Project
from app.reconstruction import reconstruct_project


def source_slide(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (160, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([12, 12, 148, 30], fill="#dbeafe")
    draw.rectangle([20, 17, 70, 24], fill="black")
    draw.line([22, 68, 140, 68], fill="#d71920", width=4)
    image.save(image_path)
    return image_path


def test_reconstruction_converts_solid_report_box_to_shape_primitive(tmp_path):
    image_path = source_slide(tmp_path)
    project = Project(
        id="primitive-shape",
        image_path=str(image_path),
        width=160,
        height=90,
        components=[
            Component(
                id="title-band",
                type="shape",
                bbox=BBox(x=12, y=12, width=136, height=18),
                source="opencv-residual",
            ),
            Component(
                id="title-text",
                type="text",
                bbox=BBox(x=20, y=17, width=50, height=7),
                text="Editable title",
                source="paddleocr",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert [primitive.kind for primitive in primitives] == ["shape", "textbox"]
    assert primitives[0].fill_color == "DBEAFE"
    assert primitives[0].source_component_id == "title-band"
    assert primitives[1].text == "Editable title"


def test_reconstruction_keeps_shape_full_when_text_overlays_most_of_it(tmp_path):
    image_path = tmp_path / "text_on_shape.png"
    image = Image.new("RGB", (160, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 140, 50], fill="#dbeafe")
    draw.rectangle([30, 24, 130, 46], fill="black")
    image.save(image_path)
    project = Project(
        id="primitive-overlaid-text",
        image_path=str(image_path),
        width=160,
        height=90,
        components=[
            Component(
                id="label-band",
                type="shape",
                bbox=BBox(x=20, y=20, width=120, height=30),
                source="opencv-residual",
            ),
            Component(
                id="label-text",
                type="text",
                bbox=BBox(x=30, y=24, width=100, height=22),
                text="Overlaid label",
                source="paddleocr",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert [primitive.kind for primitive in primitives] == ["shape", "textbox"]
    assert primitives[0].bbox == BBox(x=20, y=20, width=120, height=30)
    assert primitives[0].fill_color == "DBEAFE"
    assert primitives[1].bbox == BBox(x=30, y=24, width=100, height=22)
    assert primitives[1].text == "Overlaid label"


def test_reconstruction_keeps_complex_icon_as_picture_primitive(tmp_path):
    image_path = source_slide(tmp_path)
    project = Project(
        id="primitive-picture",
        image_path=str(image_path),
        width=160,
        height=90,
        components=[
            Component(
                id="icon",
                type="icon",
                bbox=BBox(x=100, y=30, width=34, height=34),
                source="sam3",
            )
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert len(primitives) == 1
    assert primitives[0].kind == "picture"
    assert primitives[0].source_component_id == "icon"


def test_reconstruction_converts_arrow_component_to_arrow_primitive(tmp_path):
    image_path = source_slide(tmp_path)
    project = Project(
        id="primitive-arrow",
        image_path=str(image_path),
        width=160,
        height=90,
        components=[
            Component(
                id="arrow",
                type="arrow",
                bbox=BBox(x=22, y=66, width=118, height=4),
                source="sam3",
            )
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert len(primitives) == 1
    assert primitives[0].kind == "arrow"
    assert primitives[0].line_color == "D71920"


def test_reconstruction_preserves_full_width_textured_header_as_picture(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (160, 90), "white")
    pixels = image.load()
    for y in range(22):
        for x in range(160):
            pixels[x, y] = (20 + (x * 7 + y * 3) % 70, 58 + (x * 5) % 80, 74 + (y * 11) % 70)
    image.save(image_path)
    project = Project(
        id="primitive-header",
        image_path=str(image_path),
        width=160,
        height=90,
        components=[
            Component(
                id="header",
                type="shape",
                bbox=BBox(x=0, y=0, width=160, height=22),
                source="opencv-residual",
            )
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert len(primitives) == 1
    assert primitives[0].kind == "picture"
    assert primitives[0].source_component_id == "header"


def test_reconstruction_pairs_thin_vertical_border_lines_into_rectangle_shape(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (160, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.line([20, 20, 20, 70], fill="#173f73", width=3)
    draw.line([120, 20, 120, 70], fill="#173f73", width=3)
    image.save(image_path)
    project = Project(
        id="primitive-frame",
        image_path=str(image_path),
        width=160,
        height=90,
        components=[
            Component(
                id="left-border",
                type="icon",
                bbox=BBox(x=19, y=20, width=3, height=50),
                source="opencv-residual",
            ),
            Component(
                id="right-border",
                type="icon",
                bbox=BBox(x=119, y=20, width=3, height=50),
                source="opencv-residual",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert len(primitives) == 1
    assert primitives[0].kind == "shape"
    assert primitives[0].fill_color is None
    assert primitives[0].line_color == "173F73"
    assert primitives[0].bbox == BBox(x=19, y=20, width=103, height=50)


def test_reconstruction_groups_full_border_lines_into_one_rectangle_shape(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (180, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line([30, 25, 145, 25], fill="#173f73", width=3)
    draw.line([30, 75, 145, 75], fill="#173f73", width=3)
    draw.line([30, 25, 30, 75], fill="#173f73", width=3)
    draw.line([145, 25, 145, 75], fill="#173f73", width=3)
    image.save(image_path)
    project = Project(
        id="primitive-full-frame",
        image_path=str(image_path),
        width=180,
        height=100,
        components=[
            Component(
                id="top-border",
                type="icon",
                bbox=BBox(x=30, y=24, width=115, height=3),
                source="opencv-residual",
            ),
            Component(
                id="bottom-border",
                type="icon",
                bbox=BBox(x=30, y=74, width=115, height=3),
                source="opencv-residual",
            ),
            Component(
                id="left-border",
                type="icon",
                bbox=BBox(x=29, y=25, width=3, height=50),
                source="opencv-residual",
            ),
            Component(
                id="right-border",
                type="icon",
                bbox=BBox(x=144, y=25, width=3, height=50),
                source="opencv-residual",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert len(primitives) == 1
    assert primitives[0].kind == "shape"
    assert primitives[0].fill_color is None
    assert primitives[0].line_color == "173F73"


def test_reconstruction_uses_hidden_thin_lines_to_restore_frame_shape(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (220, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.line([40, 30, 180, 30], fill="#d71920", width=2)
    draw.line([40, 110, 180, 110], fill="#d71920", width=2)
    draw.line([40, 30, 40, 110], fill="#d71920", width=2)
    draw.line([180, 30, 180, 110], fill="#d71920", width=2)
    image.save(image_path)
    project = Project(
        id="primitive-hidden-frame",
        image_path=str(image_path),
        width=220,
        height=140,
        components=[
            Component(id="top", type="shape", bbox=BBox(x=40, y=29, width=140, height=2), source="opencv-residual", hidden=True),
            Component(id="bottom", type="shape", bbox=BBox(x=40, y=109, width=140, height=2), source="opencv-residual", hidden=True),
            Component(id="left", type="shape", bbox=BBox(x=39, y=30, width=2, height=80), source="opencv-residual", hidden=True),
            Component(id="right", type="shape", bbox=BBox(x=179, y=30, width=2, height=80), source="opencv-residual", hidden=True),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert len(primitives) == 1
    assert primitives[0].kind == "shape"
    assert primitives[0].fill_color is None
    assert primitives[0].line_color == "D71920"
    assert primitives[0].bbox.width >= 140
    assert primitives[0].bbox.height >= 80


def test_reconstruction_writes_synthetic_frame_as_outline_only_shape(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (220, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 30, 180, 110], outline="#173f73", width=2)
    image.save(image_path)
    project = Project(
        id="primitive-synthetic-frame",
        image_path=str(image_path),
        width=220,
        height=140,
        components=[
            Component(
                id="synthetic-frame",
                type="shape",
                bbox=BBox(x=40, y=30, width=140, height=80),
                source="synthetic-frame-shape",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert len(primitives) == 1
    assert primitives[0].kind == "shape"
    assert primitives[0].fill_color is None
    assert primitives[0].line_color == "173F73"


def test_reconstruction_deduplicates_synthetic_frame_and_detected_rules(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (260, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 30, 220, 130], outline="#173f73", width=2)
    image.save(image_path)
    project = Project(
        id="primitive-dedup-synthetic-frame",
        image_path=str(image_path),
        width=260,
        height=160,
        components=[
            Component(
                id="left-edge",
                type="shape",
                bbox=BBox(x=40, y=30, width=2, height=100),
                source="opencv-residual",
                hidden=True,
            ),
            Component(
                id="right-edge",
                type="shape",
                bbox=BBox(x=218, y=30, width=2, height=100),
                source="opencv-residual",
                hidden=True,
            ),
            Component(
                id="synthetic-frame",
                type="shape",
                bbox=BBox(x=40, y=30, width=180, height=100),
                source="synthetic-frame-shape",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert len(primitives) == 1
    assert primitives[0].source_component_id == "synthetic-frame"


def test_reconstruction_uses_saturated_outline_color_for_pale_card_edge(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (260, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 30, 220, 130], fill="#fff8f5", outline="#d13a1a", width=2)
    image.save(image_path)
    project = Project(
        id="primitive-card-saturated-outline",
        image_path=str(image_path),
        width=260,
        height=160,
        components=[
            Component(
                id="card",
                type="shape",
                bbox=BBox(x=40, y=30, width=180, height=100),
                source="synthetic-info-card-shape",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert len(primitives) == 1
    color = primitives[0].line_color
    assert color is not None
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    assert max(red, green, blue) - min(red, green, blue) >= 80


def test_reconstruction_keeps_connected_box_and_line_as_separate_primitives(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (160, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 80, 60], fill="#dbeafe")
    draw.line([80, 40, 130, 40], fill="#d71920", width=4)
    image.save(image_path)
    project = Project(
        id="primitive-box-line",
        image_path=str(image_path),
        width=160,
        height=90,
        components=[
            Component(
                id="box",
                type="shape",
                bbox=BBox(x=20, y=20, width=60, height=40),
                source="opencv-residual",
            ),
            Component(
                id="line",
                type="icon",
                bbox=BBox(x=80, y=38, width=50, height=4),
                source="opencv-residual",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert [primitive.kind for primitive in primitives] == ["shape", "line"]
    assert primitives[0].source_component_id == "box"
    assert primitives[1].source_component_id == "line"


def test_reconstruction_splits_background_shape_from_image_on_top(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (180, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 160, 70], fill="#dbeafe")
    draw.ellipse([70, 28, 110, 68], fill="#d71920")
    image.save(image_path)
    project = Project(
        id="primitive-shape-image-stack",
        image_path=str(image_path),
        width=180,
        height=100,
        components=[
            Component(
                id="background",
                type="shape",
                bbox=BBox(x=20, y=20, width=140, height=50),
                source="opencv-residual",
            ),
            Component(
                id="image",
                type="image",
                bbox=BBox(x=70, y=28, width=40, height=40),
                source="sam3",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert [primitive.kind for primitive in primitives] == ["shape", "picture"]
    assert primitives[0].bbox == BBox(x=20, y=20, width=140, height=50)
    assert primitives[0].fill_color == "DBEAFE"
    assert primitives[1].bbox == BBox(x=70, y=28, width=40, height=40)


def test_reconstruction_splits_outlined_shape_card_into_shape_and_picture(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (240, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([24, 24, 216, 96], fill="#ffffff", outline="#2563eb", width=3)
    draw.ellipse([154, 42, 190, 78], fill="#d71920")
    image.save(image_path)
    project = Project(
        id="primitive-outlined-card-image-stack",
        image_path=str(image_path),
        width=240,
        height=140,
        components=[
            Component(
                id="card",
                type="shape",
                bbox=BBox(x=24, y=24, width=192, height=72),
                source="opencv-residual",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert [primitive.kind for primitive in primitives] == ["shape", "picture"]
    assert primitives[0].bbox == BBox(x=24, y=24, width=192, height=72)
    assert primitives[0].fill_color == "FFFFFF"
    assert primitives[0].line_color == "2563EB"
    assert 150 <= primitives[1].bbox.x <= 156
    assert primitives[1].bbox.width <= 42


def test_reconstruction_splits_oversized_picture_into_shape_and_inner_picture(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (180, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 160, 70], fill="#dbeafe")
    draw.ellipse([72, 30, 108, 66], fill="#d71920")
    image.save(image_path)
    project = Project(
        id="primitive-oversized-picture",
        image_path=str(image_path),
        width=180,
        height=100,
        components=[
            Component(
                id="oversized-image",
                type="image",
                bbox=BBox(x=20, y=20, width=140, height=50),
                source="sam3",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert [primitive.kind for primitive in primitives] == ["shape", "picture"]
    assert primitives[0].bbox == BBox(x=20, y=20, width=140, height=50)
    assert primitives[0].fill_color == "DBEAFE"
    assert 68 <= primitives[1].bbox.x <= 74
    assert 28 <= primitives[1].bbox.y <= 32
    assert primitives[1].bbox.width <= 45
    assert primitives[1].bbox.height <= 45


def test_reconstruction_excludes_text_when_splitting_oversized_picture(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (220, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 200, 80], fill="#dbeafe")
    draw.rectangle([34, 34, 100, 50], fill="black")
    draw.ellipse([146, 32, 184, 70], fill="#d71920")
    image.save(image_path)
    project = Project(
        id="primitive-oversized-picture-with-text",
        image_path=str(image_path),
        width=220,
        height=120,
        components=[
            Component(
                id="oversized-image",
                type="image",
                bbox=BBox(x=20, y=20, width=180, height=60),
                source="sam3",
            ),
            Component(
                id="text",
                type="text",
                bbox=BBox(x=34, y=34, width=66, height=16),
                text="Editable",
                source="paddleocr",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        primitives = reconstruct_project(project, source)

    assert [primitive.kind for primitive in primitives] == ["shape", "picture", "textbox"]
    assert primitives[1].bbox.x >= 140
    assert primitives[1].bbox.width <= 50
    assert primitives[2].text == "Editable"
