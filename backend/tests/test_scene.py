import base64
from io import BytesIO
from xml.etree import ElementTree

from PIL import Image, ImageDraw

from app.models import BBox, Component, Project
from app.scene import build_scene_graph, render_scene_svg


def test_scene_graph_renders_svg_with_native_rect_text_and_embedded_image(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (180, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 160, 50], fill="#dbeafe")
    draw.ellipse([120, 60, 150, 90], fill="#d71920")
    image.save(image_path)
    project = Project(
        id="scene-test",
        image_path=str(image_path),
        width=180,
        height=100,
        components=[
            Component(id="band", type="shape", bbox=BBox(x=20, y=20, width=140, height=30), source="opencv-residual"),
            Component(
                id="title",
                type="text",
                bbox=BBox(x=30, y=26, width=80, height=18),
                text="Editable",
                source="paddleocr",
            ),
            Component(id="icon", type="icon", bbox=BBox(x=120, y=60, width=30, height=30), source="sam3"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)
        svg = render_scene_svg(scene, source)

    assert [node.kind for node in scene.nodes] == ["rect", "image", "text"]
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
    assert "<rect" in svg
    assert "<text" in svg
    assert "Editable" in svg
    assert "data:image/png;base64," in svg

    root = ElementTree.fromstring(svg)
    assert root.attrib["viewBox"] == "0 0 180 100"


def test_scene_graph_erases_header_picture_text_under_editable_text(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    for x in range(200):
        draw.line([(x, 0), (x, 20)], fill=(30 + x % 30, 90, 120))
    draw.rectangle([20, 6, 80, 14], fill="black")
    image.save(image_path)
    project = Project(
        id="scene-header",
        image_path=str(image_path),
        width=200,
        height=100,
        components=[
            Component(id="header", type="shape", bbox=BBox(x=0, y=0, width=200, height=20), source="opencv-residual"),
            Component(id="title", type="text", bbox=BBox(x=20, y=6, width=60, height=8), text="Title", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)
        svg = render_scene_svg(scene, source)

    image_node = next(node for node in scene.nodes if node.kind == "image")
    assert image_node.erase_boxes
    embedded = _first_embedded_png(svg)
    assert embedded.getpixel((40, 10))[:3] != (0, 0, 0)


def test_scene_graph_erases_chart_picture_text_under_editable_text(tmp_path):
    image_path = tmp_path / "chart.png"
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 20, 170, 90], outline="#2563eb", width=2)
    draw.rectangle([50, 36, 120, 48], fill="black")
    image.save(image_path)
    project = Project(
        id="scene-chart",
        image_path=str(image_path),
        width=200,
        height=120,
        components=[
            Component(id="chart", type="chart", bbox=BBox(x=30, y=20, width=140, height=70), source="sam3"),
            Component(id="chart-label", type="text", bbox=BBox(x=50, y=36, width=70, height=12), text="Axis", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    image_node = next(node for node in scene.nodes if node.kind == "image")
    assert image_node.source_component_type == "chart"
    assert image_node.erase_boxes


def test_scene_graph_does_not_erase_text_that_only_slightly_overlaps_large_image(tmp_path):
    image_path = tmp_path / "large-image-title-overlap.png"
    image = Image.new("RGB", (300, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([170, 0, 300, 160], fill="#d71920")
    draw.rectangle([20, 6, 188, 42], fill="black")
    image.save(image_path)
    project = Project(
        id="scene-large-image-title-overlap",
        image_path=str(image_path),
        width=300,
        height=160,
        components=[
            Component(id="hero-image", type="image", bbox=BBox(x=170, y=0, width=130, height=160), source="opencv-residual"),
            Component(id="title", type="text", bbox=BBox(x=20, y=6, width=168, height=36), text="Strong Title", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    image_node = next(node for node in scene.nodes if node.kind == "image")
    text_node = next(node for node in scene.nodes if node.kind == "text")
    assert image_node.erase_boxes == []
    assert text_node.bbox.x + text_node.bbox.width <= image_node.bbox.x


def test_scene_graph_keeps_illustration_pixels_under_editable_text(tmp_path):
    image_path = tmp_path / "illustration.png"
    image = Image.new("RGB", (300, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([120, 0, 300, 160], fill="#d71920")
    draw.rectangle([190, 104, 276, 134], fill="white")
    image.save(image_path)
    project = Project(
        id="scene-large-illustration-text",
        image_path=str(image_path),
        width=300,
        height=160,
        components=[
            Component(id="hero-image", type="image", bbox=BBox(x=120, y=0, width=180, height=160), source="opencv-residual"),
            Component(id="decorative-text", type="text", bbox=BBox(x=190, y=104, width=86, height=30), text="BAM", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    image_node = next(node for node in scene.nodes if node.kind == "image")
    text_node = next(node for node in scene.nodes if node.kind == "text")
    assert text_node.text == "BAM"
    assert image_node.erase_boxes == []


def test_scene_graph_keeps_huge_stylized_image_text_as_artwork(tmp_path):
    image_path = tmp_path / "huge-stylized-text.png"
    image = Image.new("RGB", (300, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([120, 0, 300, 160], fill="#d71920")
    draw.rectangle([140, 76, 292, 154], fill="white")
    image.save(image_path)
    project = Project(
        id="huge-stylized-text",
        image_path=str(image_path),
        width=300,
        height=160,
        components=[
            Component(id="art", type="image", bbox=BBox(x=120, y=0, width=180, height=160), source="sam3"),
            Component(id="text", type="text", bbox=BBox(x=140, y=76, width=152, height=78), text="BAM", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    image_node = next(node for node in scene.nodes if node.kind == "image")
    text_node = next(node for node in scene.nodes if node.kind == "text")
    assert text_node.text == "BAM"
    assert image_node.erase_boxes == []


def test_scene_graph_keeps_complex_region_text_editable(tmp_path):
    image_path = tmp_path / "complex-banner.png"
    image = Image.new("RGB", (420, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([24, 58, 390, 118], fill="#fdfdfd", outline="#bd110d", width=2)
    draw.polygon([(24, 58), (160, 58), (140, 118), (24, 118)], fill="#d71920")
    draw.rectangle([52, 72, 138, 100], fill="white")
    draw.rectangle([190, 78, 330, 94], fill="black")
    image.save(image_path)
    project = Project(
        id="scene-complex-banner",
        image_path=str(image_path),
        width=420,
        height=140,
        components=[
            Component(id="banner", type="shape", bbox=BBox(x=24, y=58, width=366, height=60), source="opencv-residual"),
            Component(id="label", type="text", bbox=BBox(x=52, y=72, width=86, height=28), text="KEY TAKEAWAY", source="paddleocr"),
            Component(id="body", type="text", bbox=BBox(x=190, y=78, width=140, height=16), text="Editable summary", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)
        svg = render_scene_svg(scene, source)

    text_nodes = [node for node in scene.nodes if node.kind == "text" and node.text != "•"]
    image_nodes = [node for node in scene.nodes if node.kind == "image"]
    assert [node.text for node in text_nodes] == ["KEY TAKEAWAY", "Editable summary"]
    assert not [node for node in scene.nodes if node.kind == "text" and node.text == "•"]
    assert image_nodes
    assert image_nodes[0].erase_boxes
    embedded = _first_embedded_png(svg)
    assert embedded.getpixel((40, 10))[:3] == (215, 25, 32)
    assert embedded.getpixel((166, 20))[:3] != (0, 0, 0)


def test_scene_graph_keeps_dense_icon_text_list_editable(tmp_path):
    image_path = tmp_path / "dense-list.png"
    image = Image.new("RGB", (520, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([24, 28, 470, 386], outline="#d1d5db", width=2)
    for index, y in enumerate((70, 140, 210, 280)):
        draw.ellipse([42, y, 90, y + 48], fill="#111827")
        draw.rectangle([112, y + 6, 154, y + 44], fill="#facc15")
        draw.rectangle([178, y + 6, 398, y + 22], fill="black")
        if index < 3:
            draw.line([42, y + 62, 430, y + 62], fill="#d1d5db", width=1)
    image.save(image_path)
    components = []
    for index, y in enumerate((70, 140, 210, 280)):
        components.extend(
            [
                Component(id=f"icon-{index}", type="icon", bbox=BBox(x=42, y=y, width=48, height=48), source="opencv-residual"),
                Component(id=f"num-bg-{index}", type="shape", bbox=BBox(x=112, y=y + 6, width=42, height=38), source="opencv-residual"),
                Component(id=f"heading-{index}", type="text", bbox=BBox(x=178, y=y + 6, width=220, height=16), text=f"ITEM {index + 1}", source="paddleocr"),
            ]
        )
    project = Project(id="scene-dense-list", image_path=str(image_path), width=520, height=420, components=components)

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    text_nodes = [node for node in scene.nodes if node.kind == "text"]
    image_nodes = [node for node in scene.nodes if node.kind == "image"]
    rect_nodes = [node for node in scene.nodes if node.kind == "rect"]
    assert [node.text for node in text_nodes] == ["ITEM 1", "ITEM 2", "ITEM 3", "ITEM 4"]
    assert len(image_nodes) == 4
    assert len(rect_nodes) == 4


def test_scene_graph_synthesizes_consistent_number_badges_from_source(tmp_path):
    image_path = tmp_path / "number-badges.png"
    image = Image.new("RGB", (260, 520), "white")
    draw = ImageDraw.Draw(image)
    y_positions = [40, 130, 220, 310, 400]
    text_boxes = [
        BBox(x=56, y=50, width=30, height=55),
        BBox(x=50, y=140, width=47, height=62),
        BBox(x=55, y=230, width=37, height=55),
        BBox(x=51, y=320, width=45, height=64),
        BBox(x=52, y=410, width=41, height=59),
    ]
    for y in y_positions:
        draw.rectangle([50, y, 99, y + 77], fill="#facc15")
    image.save(image_path)
    project = Project(
        id="scene-number-badges",
        image_path=str(image_path),
        width=260,
        height=520,
        components=[
            Component(id=f"digit-{index}", type="text", bbox=text_boxes[index], text=str(index + 1), source="paddleocr")
            for index in range(5)
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    badge_rects = [
        node
        for node in scene.nodes
        if node.kind == "rect" and node.fill_color and node.fill_color.upper().startswith("FA")
    ]
    digit_nodes = [node for node in scene.nodes if node.kind == "text" and node.text in {"1", "2", "3", "4", "5"}]

    assert len(badge_rects) == 5
    assert len(digit_nodes) == 5
    assert {round(node.bbox.width) for node in digit_nodes} == {50}
    assert {round(node.bbox.height) for node in digit_nodes} == {78}


def test_scene_graph_does_not_synthesize_bullets_from_number_badges(tmp_path):
    image_path = tmp_path / "number-badge-body.png"
    image = Image.new("RGB", (320, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 30, 99, 107], fill="#facc15")
    draw.rectangle([68, 54, 81, 92], fill="black")
    image.save(image_path)
    project = Project(
        id="scene-number-badge-body",
        image_path=str(image_path),
        width=320,
        height=140,
        components=[
            Component(id="digit", type="text", bbox=BBox(x=56, y=42, width=30, height=55), text="1", source="paddleocr"),
            Component(id="body", type="text", bbox=BBox(x=150, y=64, width=120, height=20), text="Body copy", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    assert not [node for node in scene.nodes if node.kind == "text" and node.text == "•"]


def test_scene_graph_does_not_synthesize_bullets_inside_right_side_table(tmp_path):
    image_path = tmp_path / "table-cell.png"
    image = Image.new("RGB", (360, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([180, 40, 340, 120], outline="#d1d5db", width=1)
    draw.ellipse([198, 66, 204, 72], fill="black")
    image.save(image_path)
    project = Project(
        id="scene-table-cell-no-bullet",
        image_path=str(image_path),
        width=360,
        height=160,
        components=[
            Component(id="cell-text", type="text", bbox=BBox(x=250, y=58, width=70, height=22), text="Cell text", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    assert not [node for node in scene.nodes if node.kind == "text" and node.text == "•"]


def test_scene_graph_expands_label_text_box_to_colored_backplate(tmp_path):
    image_path = tmp_path / "label-backplate.png"
    image = Image.new("RGB", (360, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 36, 260, 88], fill="#bd0f0e")
    image.save(image_path)
    project = Project(
        id="scene-label-backplate",
        image_path=str(image_path),
        width=360,
        height=120,
        components=[
            Component(id="label-bg", type="shape", bbox=BBox(x=40, y=36, width=220, height=52), source="opencv-residual"),
            Component(id="label", type="text", bbox=BBox(x=76, y=46, width=150, height=30), text="D) KEY TAKEAWAY", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    label = next(node for node in scene.nodes if node.kind == "text" and node.text == "D) KEY TAKEAWAY")
    assert label.bbox.width > 170
    assert label.bbox.height == 30


def test_scene_graph_removes_redundant_header_subimage(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (220, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 220, 24], fill="#123456")
    image.save(image_path)
    project = Project(
        id="scene-header-duplicate",
        image_path=str(image_path),
        width=220,
        height=120,
        components=[
            Component(id="header", type="shape", bbox=BBox(x=0, y=0, width=220, height=24), source="opencv-residual"),
            Component(id="duplicate", type="image", bbox=BBox(x=2, y=2, width=216, height=12), source="sam3"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    image_nodes = [node for node in scene.nodes if node.kind == "image"]
    assert len(image_nodes) == 1
    assert image_nodes[0].source_component_id == "header"


def test_scene_graph_estimates_light_text_color_on_dark_background(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (180, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 180, 24], fill="#123456")
    draw.rectangle([24, 8, 92, 16], fill="white")
    image.save(image_path)
    project = Project(
        id="scene-text-color",
        image_path=str(image_path),
        width=180,
        height=90,
        components=[
            Component(id="header", type="shape", bbox=BBox(x=0, y=0, width=180, height=24), source="opencv-residual"),
            Component(id="title", type="text", bbox=BBox(x=18, y=4, width=82, height=18), text="Title", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)
        svg = render_scene_svg(scene, source)

    text_node = next(node for node in scene.nodes if node.kind == "text")
    red = int(text_node.text_color[0:2], 16)
    green = int(text_node.text_color[2:4], 16)
    blue = int(text_node.text_color[4:6], 16)
    assert red + green + blue > 660
    assert f'fill="#{text_node.text_color}"' in svg


def test_scene_graph_estimates_dark_text_color_on_light_label(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (180, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 150, 48], fill="#e8eef7")
    draw.rectangle([42, 28, 96, 37], fill="#123456")
    image.save(image_path)
    project = Project(
        id="scene-dark-text-color",
        image_path=str(image_path),
        width=180,
        height=90,
        components=[
            Component(id="band", type="shape", bbox=BBox(x=20, y=20, width=130, height=28), source="opencv-residual"),
            Component(id="label", type="text", bbox=BBox(x=36, y=24, width=70, height=18), text="Label", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    text_node = next(node for node in scene.nodes if node.kind == "text")
    red = int(text_node.text_color[0:2], 16)
    green = int(text_node.text_color[2:4], 16)
    blue = int(text_node.text_color[4:6], 16)
    assert red + green + blue < 260


def test_scene_graph_keeps_top_black_title_dark_on_white_background(tmp_path):
    image_path = tmp_path / "top-title.png"
    image = Image.new("RGB", (260, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([24, 8, 210, 58], fill="black")
    image.save(image_path)
    project = Project(
        id="scene-top-black-title",
        image_path=str(image_path),
        width=260,
        height=120,
        components=[
            Component(
                id="title",
                type="text",
                bbox=BBox(x=24, y=8, width=186, height=50),
                text="Title",
                source="paddleocr",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)
        svg = render_scene_svg(scene, source)

    text_node = next(node for node in scene.nodes if node.kind == "text")
    assert _color_luma(text_node.text_color) < 60

    root = ElementTree.fromstring(svg)
    text = next(item for item in root.iter() if item.tag.endswith("text"))
    assert float(text.attrib["font-size"]) >= 34


def test_scene_graph_uses_condensed_display_style_for_large_english_title(tmp_path):
    image_path = tmp_path / "display-title.png"
    Image.new("RGB", (1200, 400), "white").save(image_path)
    project = Project(
        id="scene-display-title-style",
        image_path=str(image_path),
        width=1200,
        height=400,
        components=[
            Component(
                id="title",
                type="text",
                bbox=BBox(x=20, y=8, width=840, height=92),
                text="Why Is One Punch Man So Strong?",
                source="paddleocr",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        svg = render_scene_svg(build_scene_graph(project, source), source)

    root = ElementTree.fromstring(svg)
    text = next(item for item in root.iter() if item.tag.endswith("text"))
    assert float(text.attrib["font-size"]) >= 58
    assert "Impact" in text.attrib["font-family"]
    assert text.attrib["font-weight"] == "900"


def test_scene_graph_merges_adjacent_ocr_words_on_same_title_line(tmp_path):
    image_path = tmp_path / "title-line.png"
    Image.new("RGB", (520, 120), "white").save(image_path)
    project = Project(
        id="scene-title-merge",
        image_path=str(image_path),
        width=520,
        height=120,
        components=[
            Component(id="why", type="text", bbox=BBox(x=12, y=8, width=74, height=56), text="Why Is", source="paddleocr"),
            Component(id="one", type="text", bbox=BBox(x=92, y=10, width=58, height=52), text="One", source="paddleocr"),
            Component(id="punch", type="text", bbox=BBox(x=154, y=10, width=100, height=52), text="Punch", source="paddleocr"),
            Component(id="man", type="text", bbox=BBox(x=258, y=10, width=68, height=52), text="Man", source="paddleocr"),
            Component(id="so", type="text", bbox=BBox(x=330, y=10, width=52, height=52), text="S0 S", source="paddleocr"),
            Component(id="strong", type="text", bbox=BBox(x=386, y=10, width=120, height=52), text="Strong?", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    text_nodes = [node for node in scene.nodes if node.kind == "text"]
    assert len(text_nodes) == 1
    assert text_nodes[0].text == "Why Is One Punch Man So Strong?"
    assert text_nodes[0].bbox.x == 12
    assert text_nodes[0].bbox.x + text_nodes[0].bbox.width == 506


def test_scene_graph_does_not_merge_large_number_label_with_body_text(tmp_path):
    image_path = tmp_path / "number-label.png"
    Image.new("RGB", (320, 120), "white").save(image_path)
    project = Project(
        id="scene-number-label",
        image_path=str(image_path),
        width=320,
        height=120,
        components=[
            Component(id="number", type="text", bbox=BBox(x=36, y=28, width=32, height=56), text="1", source="paddleocr"),
            Component(id="body", type="text", bbox=BBox(x=96, y=40, width=180, height=20), text="Saitama was designed", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    text_nodes = sorted([node for node in scene.nodes if node.kind == "text"], key=lambda node: node.bbox.x)
    assert [node.text for node in text_nodes] == ["1", "Saitama was designed"]


def test_scene_graph_prefers_light_text_over_shadow_on_dark_header(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (220, 100), "#0f4f55")
    draw = ImageDraw.Draw(image)
    draw.rectangle([28, 10, 104, 19], fill="#041e1e")
    draw.rectangle([24, 7, 100, 16], fill="white")
    image.save(image_path)
    project = Project(
        id="scene-shadowed-light-text",
        image_path=str(image_path),
        width=220,
        height=100,
        components=[
            Component(id="header", type="shape", bbox=BBox(x=0, y=0, width=220, height=24), source="opencv-residual"),
            Component(id="title", type="text", bbox=BBox(x=18, y=4, width=94, height=20), text="Title", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    text_node = next(node for node in scene.nodes if node.kind == "text")
    red = int(text_node.text_color[0:2], 16)
    green = int(text_node.text_color[2:4], 16)
    blue = int(text_node.text_color[4:6], 16)
    assert red + green + blue > 660


def test_scene_graph_does_not_turn_plain_dark_heading_into_background_rect(tmp_path):
    image_path = tmp_path / "plain-heading.png"
    image = Image.new("RGB", (260, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([42, 82, 198, 106], fill="black")
    image.save(image_path)
    project = Project(
        id="scene-plain-dark-heading",
        image_path=str(image_path),
        width=260,
        height=160,
        components=[
            Component(
                id="heading",
                type="text",
                bbox=BBox(x=42, y=82, width=156, height=24),
                text="BUILT AS A SATIRE",
                source="paddleocr",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    text_node = next(node for node in scene.nodes if node.kind == "text")
    assert _color_luma(text_node.text_color) < 60
    assert not [
        node
        for node in scene.nodes
        if node.kind == "rect" and node.fill_color and _color_luma(node.fill_color) < 60
    ]


def test_scene_graph_synthesizes_missing_colored_text_background_rect(tmp_path):
    image_path = tmp_path / "slide.png"
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 220, 62], fill="#eaeff7")
    draw.rectangle([62, 26, 168, 56], fill="#132a4d")
    draw.rectangle([82, 36, 148, 45], fill="white")
    image.save(image_path)
    project = Project(
        id="scene-missing-label-bg",
        image_path=str(image_path),
        width=240,
        height=120,
        components=[
            Component(id="message-band", type="shape", bbox=BBox(x=20, y=20, width=200, height=42), source="opencv-residual"),
            Component(id="label-text", type="text", bbox=BBox(x=76, y=31, width=82, height=20), text="핵심 메시지", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    rects = [node for node in scene.nodes if node.kind == "rect" and node.fill_color == "132A4D"]
    assert rects
    assert rects[0].z_index < 3000
    assert rects[0].bbox.x <= 66
    assert rects[0].bbox.x + rects[0].bbox.width >= 164


def test_scene_graph_synthesizes_info_card_backgrounds_from_icon_and_text_groups(tmp_path):
    image_path = tmp_path / "cards.png"
    image = Image.new("RGB", (560, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 84, 110], fill="#fff8f5", outline="#d03a1a", width=2)
    draw.rectangle([300, 50, 334, 110], fill="#f5f9fd", outline="#428dd5", width=2)
    image.save(image_path)
    project = Project(
        id="scene-info-cards",
        image_path=str(image_path),
        width=560,
        height=180,
        components=[
            Component(id="strength-icon-bg", type="shape", bbox=BBox(x=50, y=50, width=34, height=60), source="opencv-residual"),
            Component(id="strength-icon", type="icon", bbox=BBox(x=50, y=50, width=34, height=60), source="sam3"),
            Component(id="strength-title", type="text", bbox=BBox(x=110, y=54, width=54, height=24), text="강도", source="paddleocr"),
            Component(id="strength-desc", type="text", bbox=BBox(x=100, y=92, width=118, height=16), text="최고수온/최저염분", source="paddleocr"),
            Component(id="freq-icon-bg", type="shape", bbox=BBox(x=300, y=50, width=34, height=60), source="opencv-residual"),
            Component(id="freq-icon", type="icon", bbox=BBox(x=300, y=50, width=34, height=60), source="sam3"),
            Component(id="freq-title", type="text", bbox=BBox(x=360, y=54, width=54, height=24), text="빈도", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    card_rects = [node for node in scene.nodes if node.kind == "rect" and node.z_index == 950]
    assert len(card_rects) == 2
    assert card_rects[0].bbox.x < 50
    assert card_rects[0].bbox.x + card_rects[0].bbox.width > 210
    assert card_rects[0].bbox.x + card_rects[0].bbox.width < card_rects[1].bbox.x
    assert card_rects[0].fill_color == "FFF8F5"
    assert card_rects[0].line_color == "D03A1A"


def test_scene_graph_applies_common_korean_ocr_corrections(tmp_path):
    image_path = tmp_path / "ocr.png"
    Image.new("RGB", (220, 100), "white").save(image_path)
    project = Project(
        id="scene-ocr-corrections",
        image_path=str(image_path),
        width=220,
        height=100,
        components=[
            Component(
                id="text",
                type="text",
                bbox=BBox(x=20, y=20, width=160, height=22),
                text="핵심 매시지와 생산량 급감 시정 확인",
                source="paddleocr",
            ),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    text_node = next(node for node in scene.nodes if node.kind == "text")
    assert text_node.text == "핵심 메시지와 생산량 급감 시점 확인"


def test_scene_graph_applies_common_slide_label_ocr_corrections(tmp_path):
    image_path = tmp_path / "ocr-labels.png"
    Image.new("RGB", (360, 120), "white").save(image_path)
    project = Project(
        id="scene-label-ocr-corrections",
        image_path=str(image_path),
        width=360,
        height=120,
        components=[
            Component(id="section-b", type="text", bbox=BBox(x=20, y=20, width=240, height=24), text="B HOW IT WORKS", source="paddleocr"),
            Component(id="section-d", type="text", bbox=BBox(x=20, y=60, width=180, height=24), text="D] KEY TAKEAWAY", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    assert [node.text for node in scene.nodes if node.kind == "text"] == ["B) HOW IT WORKS", "D) KEY TAKEAWAY"]


def test_scene_graph_uses_source_crop_for_chart_assets_to_keep_annotations(tmp_path):
    image_path = tmp_path / "chart-source.png"
    image = Image.new("RGB", (140, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([42, 8, 98, 22], fill="#d71920")
    draw.line([24, 66, 120, 36], fill="#2563eb", width=2)
    image.save(image_path)
    asset_path = tmp_path / "chart-asset.png"
    Image.new("RGBA", (140, 90), (255, 255, 255, 255)).save(asset_path)
    mask_path = tmp_path / "chart-mask.png"
    mask = Image.new("L", (140, 90), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.line([24, 66, 120, 36], fill=255, width=8)
    mask.save(mask_path)
    project = Project(
        id="scene-chart-source-crop",
        image_path=str(image_path),
        width=140,
        height=90,
        components=[
            Component(
                id="chart",
                type="chart",
                bbox=BBox(x=0, y=0, width=140, height=90),
                asset_path=str(asset_path),
                mask_path=str(mask_path),
                source="sam3",
            )
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        svg = render_scene_svg(build_scene_graph(project, source), source)

    embedded = _first_embedded_png(svg)
    red, green, blue, alpha = embedded.getpixel((50, 14))
    assert alpha > 240
    assert (red, green, blue) == (215, 25, 32)


def test_scene_graph_scales_svg_text_to_available_width(tmp_path):
    image_path = tmp_path / "long-text.png"
    Image.new("RGB", (180, 80), "white").save(image_path)
    project = Project(
        id="scene-text-width",
        image_path=str(image_path),
        width=180,
        height=80,
        components=[
            Component(
                id="long-text",
                type="text",
                bbox=BBox(x=20, y=20, width=90, height=30),
                text="세부 연구 내용- 적합 어장 양식생산량",
                source="paddleocr",
            )
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        svg = render_scene_svg(build_scene_graph(project, source), source)

    root = ElementTree.fromstring(svg)
    text = next(item for item in root.iter() if item.tag.endswith("text"))
    assert float(text.attrib["font-size"]) <= 10


def test_scene_graph_shrinks_oversized_dark_label_background_to_text_width(tmp_path):
    image_path = tmp_path / "wide-label.png"
    image = Image.new("RGB", (500, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([220, 46, 370, 78], fill="#132a4d")
    image.save(image_path)
    project = Project(
        id="scene-shrink-label",
        image_path=str(image_path),
        width=500,
        height=140,
        components=[
            Component(id="wide-label-bg", type="shape", bbox=BBox(x=80, y=46, width=290, height=32), source="opencv-residual"),
            Component(id="wide-label-text", type="text", bbox=BBox(x=245, y=51, width=102, height=22), text="생산량 급감", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    rect = next(node for node in scene.nodes if node.kind == "rect" and node.fill_color == "132A4D")
    assert rect.bbox.x > 180
    assert rect.bbox.x < 245
    assert rect.bbox.x + rect.bbox.width > 347
    assert rect.bbox.width < 180


def test_scene_graph_trims_chart_image_below_overlapping_editable_title(tmp_path):
    image_path = tmp_path / "chart-title-overlap.png"
    image = Image.new("RGB", (220, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([60, 34, 160, 58], fill="#132a4d")
    draw.rectangle([40, 54, 180, 120], outline="#94a3b8", width=1)
    image.save(image_path)
    project = Project(
        id="scene-trim-chart-title",
        image_path=str(image_path),
        width=220,
        height=140,
        components=[
            Component(id="chart", type="chart", bbox=BBox(x=40, y=54, width=140, height=66), source="sam3"),
            Component(id="chart-title", type="text", bbox=BBox(x=72, y=36, width=76, height=24), text="분석 제목", source="paddleocr"),
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    image_node = next(node for node in scene.nodes if node.kind == "image")
    assert image_node.bbox.y >= 60
    assert image_node.bbox.height <= 60


def test_scene_graph_synthesizes_missing_bullet_text_from_source_pixels(tmp_path):
    image_path = tmp_path / "bullet.png"
    image = Image.new("RGB", (260, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([28, 42, 36, 50], fill="black")
    image.save(image_path)
    project = Project(
        id="scene-bullet",
        image_path=str(image_path),
        width=260,
        height=100,
        components=[
            Component(
                id="line",
                type="text",
                bbox=BBox(x=68, y=35, width=150, height=24),
                text="본문 설명 문장",
                source="paddleocr",
            )
        ],
    )

    with Image.open(image_path).convert("RGBA") as source:
        scene = build_scene_graph(project, source)

    bullet = next(node for node in scene.nodes if node.kind == "text" and node.text == "•")
    assert 24 <= bullet.bbox.x <= 38
    assert bullet.text_color == "111111"


def _first_embedded_png(svg: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    start = svg.index(prefix) + len(prefix)
    end = svg.index('"', start)
    return Image.open(BytesIO(base64.b64decode(svg[start:end]))).convert("RGBA")


def _color_luma(hex_color: str | None) -> float:
    assert hex_color is not None
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return red * 0.2126 + green * 0.7152 + blue * 0.0722
