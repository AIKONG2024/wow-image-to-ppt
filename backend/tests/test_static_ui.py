import re
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


def test_ui_has_canvas_busy_overlay():
    sources = [
        Path("backend/static/app.js").read_text(encoding="utf-8"),
        Path("frontend/src/App.jsx").read_text(encoding="utf-8"),
    ]
    styles = [
        Path("backend/static/base.css").read_text(encoding="utf-8"),
        Path("frontend/src/styles.css").read_text(encoding="utf-8"),
    ]

    for source in sources:
        assert "busyOverlay" in source
        assert "loadingSpinner" in source
        assert "waitForNextPaint" in source
        assert "컴포넌트를 분석하는 중입니다" in source
        assert "Analyzing components" in source

    for css_source in styles:
        assert re.search(r"\.canvasPane\s*\{[^}]*position:\s*relative", css_source, re.S)
        assert re.search(r"\.busyOverlay\s*\{[^}]*position:\s*absolute", css_source, re.S)
        assert re.search(r"\.busyOverlay\s*\{[^}]*background:\s*rgba\(15,\s*23,\s*42", css_source, re.S)
        assert ".loadingSpinner" in css_source
        assert "@keyframes spin" in css_source


def test_static_entrypoint_cache_busts_ui_assets():
    index_source = Path("backend/static/index.html").read_text(encoding="utf-8")
    styles_source = Path("backend/static/styles.css").read_text(encoding="utf-8")

    assert "/static/app.js?v=" in index_source
    assert "/static/styles.css?v=" in index_source
    assert "/static/base.css?v=" in styles_source
