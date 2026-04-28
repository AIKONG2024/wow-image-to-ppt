from app.analysis import _classify_visual, normalize_component_graph
from app.models import BBox, Component


def test_normalize_hides_page_scale_false_visual_parent_but_keeps_children():
    components = [
        Component(
            id="page-sized-false-chart",
            type="chart",
            bbox=BBox(x=0, y=0, width=1000, height=600),
            source="opencv-fallback",
        ),
        Component(
            id="title",
            type="text",
            bbox=BBox(x=40, y=30, width=420, height=40),
            text="Generated slide title",
            source="paddleocr",
        ),
        Component(
            id="section",
            type="shape",
            bbox=BBox(x=40, y=110, width=400, height=180),
            source="opencv-residual",
        ),
        Component(
            id="icon",
            type="icon",
            bbox=BBox(x=70, y=145, width=72, height=72),
            source="opencv-residual",
        ),
        Component(
            id="body-1",
            type="text",
            bbox=BBox(x=160, y=140, width=240, height=24),
            text="Editable bullet",
            source="paddleocr",
        ),
        Component(
            id="body-2",
            type="text",
            bbox=BBox(x=160, y=180, width=220, height=24),
            text="Editable text",
            source="paddleocr",
        ),
        Component(
            id="chart",
            type="chart",
            bbox=BBox(x=520, y=130, width=360, height=260),
            source="opencv-residual",
        ),
        Component(
            id="callout",
            type="text",
            bbox=BBox(x=560, y=430, width=300, height=32),
            text="A separate callout",
            source="paddleocr",
        ),
    ]

    normalized = normalize_component_graph(components)
    by_id = {component.id: component for component in normalized}

    assert by_id["page-sized-false-chart"].hidden
    assert not by_id["title"].hidden
    assert not by_id["section"].hidden
    assert not by_id["icon"].hidden


def test_classify_visual_marks_tall_side_illustration_as_image_not_chart():
    assert _classify_visual(width=540, height=941, image_width=1672, image_height=941) == "image"
    assert _classify_visual(width=586, height=300, image_width=1672, image_height=941) == "chart"
