from app.models import BBox, Component, ComponentPatch, Project
from app.operations import apply_component_patch


def component(component_id, component_type, x, y, width, height, text=None):
    return Component(
        id=component_id,
        type=component_type,
        bbox=BBox(x=x, y=y, width=width, height=height),
        text=text,
        source="test",
    )


def project_with_components(components):
    return Project(
        id="project-test",
        image_path="source.png",
        width=400,
        height=225,
        components=components,
    )


def test_merge_text_components_keeps_editable_text_component():
    project = project_with_components(
        [
            component("title", "text", 10, 20, 80, 14, "Hello"),
            component("subtitle", "text", 10, 40, 110, 14, "World"),
        ]
    )

    result = apply_component_patch(
        project,
        ComponentPatch(operation="merge", component_ids=["title", "subtitle"]),
    )

    merged = next(item for item in result.components if item.source == "user-merge")
    assert merged.type == "text"
    assert merged.text == "Hello\nWorld"
    assert merged.bbox == BBox(x=10, y=20, width=110, height=34)
    assert all(item.hidden for item in result.components if item.id in {"title", "subtitle"})


def test_merge_with_picture_component_flattens_to_image_component():
    project = project_with_components(
        [
            component("label", "text", 10, 20, 80, 14, "Figure 1"),
            component("icon", "icon", 100, 16, 30, 30),
        ]
    )

    result = apply_component_patch(
        project,
        ComponentPatch(operation="merge", component_ids=["label", "icon"]),
    )

    merged = next(item for item in result.components if item.source == "user-merge")
    assert merged.type == "image"
    assert merged.text is None
    assert merged.bbox == BBox(x=10, y=16, width=120, height=30)
    assert all(item.hidden for item in result.components if item.id in {"label", "icon"})
