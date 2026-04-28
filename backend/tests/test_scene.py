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


def test_scene_graph_marks_header_picture_text_regions_for_erasing(tmp_path):
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
    assert image_node.erase_boxes[0] == BBox(x=20, y=6, width=60, height=8)
    embedded = _first_embedded_png(svg)
    assert embedded.getpixel((40, 10))[:3] != (0, 0, 0)


def test_scene_graph_does_not_erase_text_inside_chart_pictures(tmp_path):
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
    assert image_node.erase_boxes == []


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
