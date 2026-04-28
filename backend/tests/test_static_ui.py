from pathlib import Path


def test_component_overlay_uses_explicit_z_order_for_canvas_selection():
    source = Path("backend/static/app.js").read_text(encoding="utf-8")

    assert "orderedVisible" in source
    assert "componentZIndex(component)" in source
    assert "function componentDrawOrder" in source


def test_static_ui_has_exploded_component_inspection_view():
    app_source = Path("backend/static/app.js").read_text(encoding="utf-8")
    css_source = Path("backend/static/base.css").read_text(encoding="utf-8")

    assert "viewMode" in app_source
    assert "Exploded" in app_source
    assert "explodedGrid" in app_source
    assert "function componentCropImageStyle" in app_source
    assert "function componentInspectionOrder" in app_source
    assert ".explodedGrid" in css_source
    assert ".componentCard" in css_source


def test_react_ui_has_exploded_component_inspection_view():
    app_source = Path("frontend/src/App.jsx").read_text(encoding="utf-8")
    css_source = Path("frontend/src/styles.css").read_text(encoding="utf-8")

    assert "viewMode" in app_source
    assert "Exploded" in app_source
    assert "explodedGrid" in app_source
    assert "function componentCropImageStyle" in app_source
    assert "function componentInspectionOrder" in app_source
    assert ".explodedGrid" in css_source
    assert ".componentCard" in css_source


def test_ui_has_korean_english_language_switch():
    sources = [
        Path("backend/static/app.js").read_text(encoding="utf-8"),
        Path("frontend/src/App.jsx").read_text(encoding="utf-8"),
    ]
    styles = [
        Path("backend/static/base.css").read_text(encoding="utf-8"),
        Path("frontend/src/styles.css").read_text(encoding="utf-8"),
    ]

    for source in sources:
        assert "WOW Image to PPT" in source
        assert "wow-image-to-ppt-language" in source
        assert "languageSwitch" in source
        assert "분석 실행" in source
        assert "Run analysis" in source

    for css_source in styles:
        assert ".topbarActions" in css_source
        assert ".languageSwitch button" in css_source
