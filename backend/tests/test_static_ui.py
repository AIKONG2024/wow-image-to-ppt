from pathlib import Path


def test_component_overlay_uses_explicit_z_order_for_canvas_selection():
    source = Path("backend/static/app.js").read_text(encoding="utf-8")

    assert "orderedVisible" in source
    assert "componentZIndex(component)" in source
    assert "function componentDrawOrder" in source
