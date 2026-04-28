import importlib.machinery
import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.analysis import Analyzer
from app.analysis import _create_paddle_ocr
from app.analysis import _containment_ratio
from app.analysis import _dedupe_components
from app.analysis import _ensure_local_ai_cache
from app.analysis import _refine_chart_component_bboxes
from app.analysis import _split_layered_visual_components
from app.analysis import normalize_component_graph
from app.models import BBox, Component, Project
from app.settings import Settings
from app.storage import ProjectStore


def image_project(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (160, 90), "white").save(image_path)
    settings = Settings(data_dir=tmp_path / "data")
    store = ProjectStore(settings)
    project = Project(id="project-test", image_path=str(image_path), width=160, height=90)
    return project, store


def install_fake_module(monkeypatch, name, module):
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    monkeypatch.setitem(sys.modules, name, module)


def test_ai_cache_overrides_user_home_for_paddle(monkeypatch):
    monkeypatch.setenv("HOME", r"C:\Users\Example")
    monkeypatch.setenv("USERPROFILE", r"C:\Users\Example")

    _ensure_local_ai_cache()

    cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache"
    assert os.environ["HOME"] == str(cache_dir / "home")
    assert os.environ["USERPROFILE"] == str(cache_dir / "home")
    assert os.environ["PADDLE_HOME"] == str(cache_dir / "paddle")


def test_paddleocr_v3_predict_results_become_text_components(monkeypatch, tmp_path):
    monkeypatch.setenv("PPT_AGENT_PADDLEOCR_IN_PROCESS", "1")
    fake_paddleocr = types.ModuleType("paddleocr")

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            if "use_angle_cls" in kwargs or "show_log" in kwargs:
                raise TypeError("legacy PaddleOCR arguments are not accepted")

        def predict(self, input):
            return [
                types.SimpleNamespace(
                    res={
                        "rec_polys": [
                            [[10, 20], [110, 20], [110, 44], [10, 44]],
                        ],
                        "rec_texts": ["기후변화"],
                        "rec_scores": [0.98],
                    }
                )
            ]

    fake_paddleocr.PaddleOCR = FakePaddleOCR
    install_fake_module(monkeypatch, "paddleocr", fake_paddleocr)

    project, store = image_project(tmp_path)
    analyzed = Analyzer().analyze(project, store)

    text_components = [component for component in analyzed.components if component.type == "text"]
    assert len(text_components) == 1
    assert text_components[0].text == "기후변화"
    assert text_components[0].source == "paddleocr"


def test_paddleocr_constructor_falls_back_to_cpu_after_gpu_runtime_error(monkeypatch):
    monkeypatch.setenv("PPT_AGENT_PADDLEOCR_DEVICE", "gpu")
    fake_paddleocr = types.ModuleType("paddleocr")
    attempts = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            attempts.append(kwargs.get("device"))
            if kwargs.get("device") == "gpu":
                raise OSError("cuDNN failed")
            self.kwargs = kwargs

    fake_paddleocr.PaddleOCR = FakePaddleOCR
    install_fake_module(monkeypatch, "paddleocr", fake_paddleocr)

    ocr = _create_paddle_ocr()

    assert attempts[:2] == ["gpu", "cpu"]
    assert ocr.kwargs["device"] == "cpu"


def test_sam3_numpy_outputs_become_visual_components(monkeypatch, tmp_path):
    fake_sam3 = types.ModuleType("sam3")
    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_processor_module = types.ModuleType("sam3.model.sam3_image_processor")
    fake_model_module = types.ModuleType("sam3.model")

    def build_sam3_image_model():
        return object()

    class FakeProcessor:
        def __init__(self, model):
            self.model = model

        def set_image(self, image):
            return {"image": image.size}

        def set_text_prompt(self, state, prompt):
            if prompt != "icon":
                return {
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "masks": np.empty((0, 90, 160), dtype=np.uint8),
                    "scores": np.empty((0,), dtype=np.float32),
                }
            mask = np.zeros((90, 160), dtype=np.uint8)
            mask[20:70, 10:60] = 1
            return {
                "boxes": np.array([[10, 20, 60, 70]], dtype=np.float32),
                "masks": np.array([mask]),
                "scores": np.array([0.91], dtype=np.float32),
            }

    fake_model_builder.build_sam3_image_model = build_sam3_image_model
    fake_processor_module.Sam3Processor = FakeProcessor

    install_fake_module(monkeypatch, "sam3", fake_sam3)
    install_fake_module(monkeypatch, "sam3.model", fake_model_module)
    install_fake_module(monkeypatch, "sam3.model_builder", fake_model_builder)
    install_fake_module(monkeypatch, "sam3.model.sam3_image_processor", fake_processor_module)

    project, store = image_project(tmp_path)
    analyzed = Analyzer().analyze(project, store)

    sam_components = [component for component in analyzed.components if component.source == "sam3"]
    assert len(sam_components) == 1
    assert sam_components[0].type == "icon"
    assert sam_components[0].mask_path is not None


def test_sam3_cuda_inference_uses_bfloat16_autocast(monkeypatch, tmp_path):
    monkeypatch.setenv("PPT_AGENT_DISABLE_PADDLEOCR", "1")
    fake_torch = types.ModuleType("torch")
    fake_sam3 = types.ModuleType("sam3")
    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_processor_module = types.ModuleType("sam3.model.sam3_image_processor")
    fake_model_module = types.ModuleType("sam3.model")
    active_autocast = {"depth": 0}
    calls = []
    set_image_depths = []
    prompt_depths = []

    class FakeAutocast:
        def __enter__(self):
            active_autocast["depth"] += 1

        def __exit__(self, exc_type, exc, traceback):
            active_autocast["depth"] -= 1

    def autocast(device_type, dtype):
        calls.append((device_type, dtype))
        return FakeAutocast()

    fake_torch.bfloat16 = "bfloat16"
    fake_torch.autocast = autocast
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )

    def build_sam3_image_model():
        return object()

    class FakeProcessor:
        def __init__(self, model):
            self.model = model

        def set_image(self, image):
            set_image_depths.append(active_autocast["depth"])
            return {"image": image.size}

        def set_text_prompt(self, state, prompt):
            prompt_depths.append(active_autocast["depth"])
            if prompt != "icon":
                return {
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "masks": np.empty((0, 90, 160), dtype=np.uint8),
                    "scores": np.empty((0,), dtype=np.float32),
                }
            mask = np.zeros((90, 160), dtype=np.uint8)
            mask[20:70, 10:60] = 1
            return {
                "boxes": np.array([[10, 20, 60, 70]], dtype=np.float32),
                "masks": np.array([mask]),
                "scores": np.array([0.91], dtype=np.float32),
            }

    fake_model_builder.build_sam3_image_model = build_sam3_image_model
    fake_processor_module.Sam3Processor = FakeProcessor

    install_fake_module(monkeypatch, "torch", fake_torch)
    install_fake_module(monkeypatch, "sam3", fake_sam3)
    install_fake_module(monkeypatch, "sam3.model", fake_model_module)
    install_fake_module(monkeypatch, "sam3.model_builder", fake_model_builder)
    install_fake_module(monkeypatch, "sam3.model.sam3_image_processor", fake_processor_module)

    project, store = image_project(tmp_path)
    Analyzer().analyze(project, store)

    assert calls
    assert all(call == ("cuda", "bfloat16") for call in calls)
    assert set_image_depths == [1]
    assert prompt_depths
    assert all(depth == 1 for depth in prompt_depths)


def test_analysis_adds_residual_components_for_unclaimed_visual_regions(monkeypatch, tmp_path):
    monkeypatch.setenv("PPT_AGENT_DISABLE_PADDLEOCR", "1")
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (160, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([12, 12, 148, 30], fill="#dbeafe")
    draw.ellipse([105, 45, 135, 75], fill="#d71920")
    image.save(image_path)

    fake_sam3 = types.ModuleType("sam3")
    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_processor_module = types.ModuleType("sam3.model.sam3_image_processor")
    fake_model_module = types.ModuleType("sam3.model")

    def build_sam3_image_model():
        return object()

    class FakeProcessor:
        def __init__(self, model):
            self.model = model

        def set_image(self, image):
            return {"image": image.size}

        def set_text_prompt(self, state, prompt):
            if prompt != "icon":
                return {
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "masks": np.empty((0, 90, 160), dtype=np.uint8),
                    "scores": np.empty((0,), dtype=np.float32),
                }
            mask = np.zeros((90, 160), dtype=np.uint8)
            mask[45:75, 105:135] = 1
            return {
                "boxes": np.array([[105, 45, 135, 75]], dtype=np.float32),
                "masks": np.array([mask]),
                "scores": np.array([0.92], dtype=np.float32),
            }

    fake_model_builder.build_sam3_image_model = build_sam3_image_model
    fake_processor_module.Sam3Processor = FakeProcessor

    install_fake_module(monkeypatch, "sam3", fake_sam3)
    install_fake_module(monkeypatch, "sam3.model", fake_model_module)
    install_fake_module(monkeypatch, "sam3.model_builder", fake_model_builder)
    install_fake_module(monkeypatch, "sam3.model.sam3_image_processor", fake_processor_module)

    store = ProjectStore(Settings(data_dir=tmp_path / "data"))
    project = Project(id="project-test", image_path=str(image_path), width=160, height=90)
    analyzed = Analyzer().analyze(project, store)

    residual_components = [component for component in analyzed.components if component.source == "opencv-residual"]
    assert residual_components
    assert any(component.asset_path and Path(component.asset_path).exists() for component in residual_components)
    assert any(
        component.bbox.x <= 14
        and component.bbox.y <= 14
        and component.bbox.x + component.bbox.width >= 146
        and component.bbox.y + component.bbox.height >= 28
        for component in residual_components
    )


def test_residual_shape_asset_preserves_overlaid_text_pixels(tmp_path):
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (160, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([12, 12, 148, 30], fill="#dbeafe")
    draw.rectangle([45, 17, 95, 24], fill="black")
    image.save(image_path)

    store = ProjectStore(Settings(data_dir=tmp_path / "data"))
    text_component = Component(
        id="title-text",
        type="text",
        bbox=BBox(x=45, y=17, width=50, height=7),
        text="핵심 메시지",
        source="paddleocr",
    )

    components = Analyzer()._residual_visual_components(
        image_path,
        store,
        "project-test",
        [text_component],
    )

    title_band = next(
        component
        for component in components
        if component.source == "opencv-residual"
        and component.bbox.x <= 14
        and component.bbox.y <= 14
        and component.bbox.x + component.bbox.width >= 146
        and component.bbox.y + component.bbox.height >= 28
    )
    assert title_band.asset_path

    with Image.open(title_band.asset_path).convert("RGBA") as asset:
        local_x = int(50 - title_band.bbox.x)
        local_y = int(20 - title_band.bbox.y)
        red, green, blue, alpha = asset.getpixel((local_x, local_y))

    assert alpha > 0
    assert (red, green, blue) == (0, 0, 0)


def test_dedupe_keeps_overlapping_image_and_shape_components():
    shape = Component(
        id="background-shape",
        type="shape",
        bbox=BBox(x=10, y=10, width=120, height=50),
        source="opencv-residual",
    )
    image = Component(
        id="foreground-image",
        type="image",
        bbox=BBox(x=10, y=10, width=120, height=50),
        source="sam3",
    )

    components = _dedupe_components([shape, image])

    assert {component.id for component in components} == {"background-shape", "foreground-image"}


def test_dedupe_keeps_nested_same_type_visual_when_it_is_a_child():
    container = Component(
        id="container-image",
        type="image",
        bbox=BBox(x=20, y=20, width=180, height=70),
        source="sam3",
    )
    child = Component(
        id="child-image",
        type="image",
        bbox=BBox(x=140, y=36, width=34, height=34),
        source="sam3",
    )

    components = _dedupe_components([container, child])

    assert {component.id for component in components} == {"container-image", "child-image"}


def test_analysis_splits_oversized_visual_component_into_shape_and_inner_image(tmp_path):
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (220, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 200, 80], fill="#dbeafe")
    draw.ellipse([146, 32, 184, 70], fill="#d71920")
    image.save(image_path)
    components = [
        Component(
            id="oversized-image",
            type="image",
            bbox=BBox(x=20, y=20, width=180, height=60),
            source="sam3",
        )
    ]

    split_components = _split_layered_visual_components(image_path, components)

    assert [component.type for component in split_components] == ["shape", "image"]
    assert split_components[0].bbox == BBox(x=20, y=20, width=180, height=60)
    assert split_components[0].source == "sam3-layer-shape"
    assert split_components[1].source == "sam3-layer-image"
    assert split_components[1].bbox.x >= 140
    assert split_components[1].bbox.width <= 50


def test_analysis_splits_outlined_shape_card_into_shape_and_inner_image(tmp_path):
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (240, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([24, 24, 216, 96], fill="#ffffff", outline="#2563eb", width=3)
    draw.ellipse([154, 42, 190, 78], fill="#d71920")
    image.save(image_path)
    components = [
        Component(
            id="card",
            type="shape",
            bbox=BBox(x=24, y=24, width=192, height=72),
            source="opencv-residual",
        )
    ]

    split_components = _split_layered_visual_components(image_path, components)

    assert [component.type for component in split_components] == ["shape", "image"]
    assert split_components[0].bbox == BBox(x=24, y=24, width=192, height=72)
    assert split_components[0].source == "opencv-residual-layer-shape"
    assert split_components[1].source == "opencv-residual-layer-image"
    assert 150 <= split_components[1].bbox.x <= 156
    assert split_components[1].bbox.width <= 42


def test_normalize_graph_keeps_chart_text_editable_while_hiding_visual_fragments():
    components = [
        Component(id="chart", type="chart", bbox=BBox(x=20, y=20, width=180, height=90), source="sam3"),
        Component(id="axis", type="shape", bbox=BBox(x=30, y=95, width=150, height=3), source="opencv-residual"),
        Component(id="series", type="icon", bbox=BBox(x=44, y=40, width=90, height=28), source="opencv-residual"),
        Component(
            id="axis-text",
            type="text",
            bbox=BBox(x=54, y=100, width=40, height=10),
            text="2018",
            source="paddleocr",
        ),
        Component(
            id="outside-text",
            type="text",
            bbox=BBox(x=20, y=130, width=80, height=12),
            text="outside",
            source="paddleocr",
        ),
    ]

    normalized = normalize_component_graph(components)

    by_id = {component.id: component for component in normalized}
    assert by_id["chart"].hidden is False
    assert by_id["axis"].hidden is True
    assert by_id["series"].hidden is True
    assert by_id["axis-text"].hidden is False
    assert by_id["outside-text"].hidden is False


def test_normalize_graph_prefers_inner_plot_over_oversized_chart_panel():
    components = [
        Component(id="panel-chart", type="chart", bbox=BBox(x=20, y=20, width=260, height=180), source="sam3"),
        Component(id="plot-chart", type="chart", bbox=BBox(x=60, y=58, width=180, height=72), source="sam3"),
        Component(id="card-icon", type="icon", bbox=BBox(x=52, y=146, width=32, height=32), source="sam3"),
        Component(id="card-label", type="text", bbox=BBox(x=92, y=150, width=70, height=18), text="강도", source="paddleocr"),
        Component(id="axis-text", type="text", bbox=BBox(x=86, y=112, width=34, height=12), text="2020", source="paddleocr"),
    ]

    normalized = normalize_component_graph(components)

    by_id = {component.id: component for component in normalized}
    assert by_id["panel-chart"].hidden is True
    assert by_id["plot-chart"].hidden is False
    assert by_id["axis-text"].hidden is False
    assert by_id["card-icon"].hidden is False
    assert by_id["card-label"].hidden is False


def test_refine_chart_bbox_trims_frame_and_overlapping_title_shape(tmp_path):
    image_path = tmp_path / "chart-frame.png"
    image = Image.new("RGB", (260, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 24, 230, 150], outline="#173f73", width=2)
    draw.rectangle([80, 20, 190, 44], fill="#132a4d")
    draw.line([62, 126, 210, 70], fill="#2563eb", width=2)
    draw.line([58, 126, 214, 126], fill="#777777", width=1)
    draw.line([58, 58, 58, 126], fill="#777777", width=1)
    image.save(image_path)
    components = [
        Component(id="title-shape", type="shape", bbox=BBox(x=80, y=20, width=110, height=24), source="opencv-residual"),
        Component(id="title-text", type="text", bbox=BBox(x=96, y=24, width=78, height=18), text="분석 제목", source="paddleocr"),
        Component(id="chart", type="chart", bbox=BBox(x=30, y=24, width=200, height=126), source="sam3"),
    ]

    _refine_chart_component_bboxes(image_path, components)

    chart = next(component for component in components if component.id == "chart")
    assert chart.bbox.x > 30
    assert chart.bbox.x <= 58
    assert chart.bbox.y > 44
    assert chart.bbox.x + chart.bbox.width < 230
    assert chart.bbox.y + chart.bbox.height <= 150


def test_refine_chart_bbox_uses_synthetic_frame_when_chart_was_overtrimmed(tmp_path):
    image_path = tmp_path / "chart-frame.png"
    image = Image.new("RGB", (260, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 24, 230, 150], outline="#173f73", width=2)
    draw.rectangle([80, 20, 190, 44], fill="#132a4d")
    draw.text((42, 82), "30", fill="#111111")
    draw.line([62, 126, 210, 70], fill="#2563eb", width=2)
    draw.line([58, 126, 214, 126], fill="#777777", width=1)
    draw.line([58, 58, 58, 126], fill="#777777", width=1)
    image.save(image_path)
    components = [
        Component(id="frame", type="shape", bbox=BBox(x=30, y=24, width=200, height=126), source="synthetic-frame-shape"),
        Component(id="title-shape", type="shape", bbox=BBox(x=80, y=20, width=110, height=24), source="opencv-residual"),
        Component(id="title-text", type="text", bbox=BBox(x=96, y=24, width=78, height=18), text="분석 제목", source="paddleocr"),
        Component(id="chart", type="chart", bbox=BBox(x=86, y=58, width=128, height=68), source="sam3"),
    ]

    _refine_chart_component_bboxes(image_path, components)

    chart = next(component for component in components if component.id == "chart")
    assert chart.bbox.x <= 58
    assert chart.bbox.y > 44


def test_refine_chart_bbox_uses_panel_until_info_cards_to_preserve_formula_label(tmp_path):
    image_path = tmp_path / "chart-panel.png"
    image = Image.new("RGB", (360, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 320, 220], outline="#d71920", width=2)
    draw.rectangle([118, 58, 262, 80], fill="#fff4ed", outline="#e48a63", width=1)
    draw.text((136, 62), "극한값 지표", fill="#d71920")
    draw.line([92, 154, 292, 112], fill="#2563eb", width=2)
    draw.line([88, 176, 300, 176], fill="#777777", width=1)
    draw.rectangle([74, 186, 160, 214], fill="#fff8f5", outline="#d13a1a", width=1)
    image.save(image_path)
    components = [
        Component(id="panel", type="shape", bbox=BBox(x=40, y=40, width=280, height=180), source="synthetic-chart-panel-shape"),
        Component(id="card", type="shape", bbox=BBox(x=74, y=186, width=86, height=28), source="synthetic-info-card-shape"),
        Component(id="chart", type="chart", bbox=BBox(x=88, y=100, width=212, height=78), source="sam3"),
    ]

    _refine_chart_component_bboxes(image_path, components)

    chart = next(component for component in components if component.id == "chart")
    assert chart.bbox.y <= 58
    assert chart.bbox.y + chart.bbox.height < 186


def test_refine_chart_bbox_restores_title_label_above_refined_chart(tmp_path):
    image_path = tmp_path / "chart-title.png"
    image = Image.new("RGB", (360, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 320, 220], outline="#d71920", width=2)
    draw.rectangle([132, 38, 238, 66], fill="#d71920")
    draw.text((148, 44), "분석 제목", fill="white")
    draw.line([92, 154, 292, 112], fill="#2563eb", width=2)
    draw.rectangle([74, 186, 160, 214], fill="#fff8f5", outline="#d13a1a", width=1)
    image.save(image_path)
    components = [
        Component(id="panel", type="shape", bbox=BBox(x=40, y=40, width=280, height=180), source="synthetic-chart-panel-shape"),
        Component(id="card", type="shape", bbox=BBox(x=74, y=186, width=86, height=28), source="synthetic-info-card-shape"),
        Component(id="title-shape", type="shape", bbox=BBox(x=132, y=38, width=106, height=28), source="opencv-residual", hidden=True),
        Component(id="title-text", type="text", bbox=BBox(x=148, y=44, width=74, height=18), text="분석 제목", source="paddleocr", hidden=True),
        Component(id="chart", type="chart", bbox=BBox(x=88, y=100, width=212, height=78), source="sam3"),
    ]

    _refine_chart_component_bboxes(image_path, components)

    assert next(component for component in components if component.id == "title-shape").hidden is False
    assert next(component for component in components if component.id == "title-text").hidden is False


def test_refine_chart_bbox_keeps_internal_axis_ocr_editable_after_refinement(tmp_path):
    image_path = tmp_path / "chart-axis-text.png"
    image = Image.new("RGB", (260, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 24, 230, 150], outline="#173f73", width=2)
    draw.text((72, 132), "2010", fill="#111111")
    draw.line([62, 126, 210, 70], fill="#2563eb", width=2)
    image.save(image_path)
    components = [
        Component(id="frame", type="shape", bbox=BBox(x=30, y=24, width=200, height=126), source="synthetic-frame-shape"),
        Component(id="axis-text", type="text", bbox=BBox(x=72, y=132, width=38, height=16), text="2010", source="paddleocr"),
        Component(id="chart", type="chart", bbox=BBox(x=86, y=58, width=128, height=68), source="sam3"),
    ]

    _refine_chart_component_bboxes(image_path, components)

    assert next(component for component in components if component.id == "axis-text").hidden is False


def test_normalize_graph_synthesizes_panel_shape_from_hidden_chart_parent():
    components = [
        Component(id="panel-chart", type="chart", bbox=BBox(x=20, y=20, width=300, height=220), source="sam3"),
        Component(id="plot-chart", type="chart", bbox=BBox(x=68, y=58, width=190, height=82), source="sam3"),
        Component(id="card-icon", type="icon", bbox=BBox(x=58, y=174, width=32, height=32), source="sam3"),
        Component(id="card-label", type="text", bbox=BBox(x=96, y=180, width=70, height=18), text="강도", source="paddleocr"),
    ]

    normalized = normalize_component_graph(components)

    by_id = {component.id: component for component in normalized}
    panel_shapes = [component for component in normalized if component.source == "synthetic-chart-panel-shape"]
    assert by_id["panel-chart"].hidden is True
    assert len(panel_shapes) == 1
    assert panel_shapes[0].type == "shape"
    assert panel_shapes[0].hidden is False
    assert panel_shapes[0].bbox == BBox(x=20, y=20, width=300, height=220)


def test_normalize_graph_expands_inner_chart_to_labels_but_not_cards():
    components = [
        Component(id="panel-chart", type="chart", bbox=BBox(x=20, y=20, width=300, height=220), source="sam3"),
        Component(id="plot-chart", type="chart", bbox=BBox(x=82, y=74, width=190, height=72), source="sam3"),
        Component(id="chart-title-bg", type="shape", bbox=BBox(x=74, y=42, width=210, height=24), source="opencv-residual"),
        Component(id="chart-title", type="text", bbox=BBox(x=96, y=46, width=150, height=18), text="극한값 지표", source="paddleocr"),
        Component(id="y-tick", type="text", bbox=BBox(x=52, y=96, width=18, height=14), text="2", source="paddleocr"),
        Component(id="x-label", type="text", bbox=BBox(x=102, y=154, width=34, height=14), text="2020", source="paddleocr"),
        Component(id="caption", type="text", bbox=BBox(x=92, y=174, width=160, height=16), text="<극한 이벤트>", source="paddleocr"),
        Component(id="card-icon", type="icon", bbox=BBox(x=58, y=204, width=32, height=32), source="sam3"),
        Component(id="card-label", type="text", bbox=BBox(x=96, y=208, width=70, height=18), text="강도", source="paddleocr"),
    ]

    normalized = normalize_component_graph(components)

    by_id = {component.id: component for component in normalized}
    assert by_id["panel-chart"].hidden is True
    assert by_id["plot-chart"].hidden is False
    assert by_id["plot-chart"].bbox.y <= 42
    assert by_id["plot-chart"].bbox.y + by_id["plot-chart"].bbox.height >= 188
    assert by_id["plot-chart"].bbox.y + by_id["plot-chart"].bbox.height < by_id["card-icon"].bbox.y
    assert by_id["chart-title"].hidden is False
    assert by_id["x-label"].hidden is False
    assert by_id["caption"].hidden is False
    assert by_id["card-icon"].hidden is False
    assert by_id["card-label"].hidden is False


def test_normalize_graph_synthesizes_frame_shape_from_hidden_border_fragments():
    components = [
        Component(id="chart", type="chart", bbox=BBox(x=40, y=35, width=140, height=66), source="sam3"),
        Component(id="top", type="shape", bbox=BBox(x=30, y=29, width=170, height=3), source="opencv-residual", hidden=True),
        Component(id="bottom", type="shape", bbox=BBox(x=30, y=109, width=170, height=3), source="opencv-residual", hidden=True),
        Component(id="left", type="shape", bbox=BBox(x=29, y=30, width=3, height=80), source="opencv-residual", hidden=True),
        Component(id="right", type="shape", bbox=BBox(x=199, y=30, width=3, height=80), source="opencv-residual", hidden=True),
    ]

    normalized = normalize_component_graph(components)

    frames = [component for component in normalized if component.source == "synthetic-frame-shape"]
    assert len(frames) == 1
    assert frames[0].type == "shape"
    assert frames[0].hidden is False
    assert frames[0].bbox.x <= 30
    assert frames[0].bbox.y <= 30
    assert frames[0].bbox.x + frames[0].bbox.width >= 200
    assert frames[0].bbox.y + frames[0].bbox.height >= 110


def test_normalize_graph_synthesizes_info_card_shape_from_icon_and_text():
    components = [
        Component(id="icon-bg", type="shape", bbox=BBox(x=40, y=50, width=36, height=62), source="opencv-residual"),
        Component(id="icon", type="icon", bbox=BBox(x=40, y=50, width=36, height=62), source="sam3"),
        Component(id="title", type="text", bbox=BBox(x=98, y=54, width=58, height=24), text="강도", source="paddleocr"),
        Component(id="desc", type="text", bbox=BBox(x=92, y=88, width=124, height=18), text="최고수온/최저염분", source="paddleocr"),
        Component(id="next-icon-bg", type="shape", bbox=BBox(x=260, y=50, width=36, height=62), source="opencv-residual"),
        Component(id="next-icon", type="icon", bbox=BBox(x=260, y=50, width=36, height=62), source="sam3"),
        Component(id="next-title", type="text", bbox=BBox(x=318, y=54, width=58, height=24), text="빈도", source="paddleocr"),
        Component(id="next-desc", type="text", bbox=BBox(x=312, y=88, width=92, height=18), text="고수온·저염", source="paddleocr"),
    ]

    normalized = normalize_component_graph(components)

    cards = [component for component in normalized if component.source == "synthetic-info-card-shape"]
    assert len(cards) == 2
    assert cards[0].type == "shape"
    assert cards[0].hidden is False
    assert cards[0].bbox.x < 40
    assert cards[0].bbox.x + cards[0].bbox.width > 216
    assert cards[0].bbox.x + cards[0].bbox.width < cards[1].bbox.x


def test_normalize_graph_expands_left_padding_for_tall_narrow_card_icon():
    components = [
        Component(id="thermo-bg", type="shape", bbox=BBox(x=100, y=50, width=24, height=64), source="opencv-residual"),
        Component(id="thermo-icon", type="icon", bbox=BBox(x=100, y=50, width=24, height=64), source="sam3"),
        Component(id="title", type="text", bbox=BBox(x=154, y=54, width=58, height=24), text="강도", source="paddleocr"),
        Component(
            id="description",
            type="text",
            bbox=BBox(x=150, y=90, width=124, height=18),
            text="최고수온/최저염분",
            source="paddleocr",
        ),
    ]

    normalized = normalize_component_graph(components)

    cards = [component for component in normalized if component.source == "synthetic-info-card-shape"]
    assert len(cards) == 1
    assert cards[0].bbox.x <= 65
    assert cards[0].bbox.x + cards[0].bbox.width >= 282


def test_normalize_graph_synthesizes_info_card_inside_larger_panel_shape():
    components = [
        Component(
            id="panel",
            type="shape",
            bbox=BBox(x=20, y=20, width=420, height=180),
            source="synthetic-chart-panel-shape",
        ),
        Component(id="icon-bg", type="shape", bbox=BBox(x=60, y=90, width=36, height=62), source="opencv-residual"),
        Component(id="icon", type="icon", bbox=BBox(x=60, y=90, width=36, height=62), source="sam3"),
        Component(id="title", type="text", bbox=BBox(x=118, y=94, width=58, height=24), text="강도", source="paddleocr"),
        Component(id="desc", type="text", bbox=BBox(x=112, y=128, width=124, height=18), text="최고수온/최저염분", source="paddleocr"),
    ]

    normalized = normalize_component_graph(components)

    cards = [component for component in normalized if component.source == "synthetic-info-card-shape"]
    assert len(cards) == 1
    assert cards[0].bbox.x < 60
    assert cards[0].bbox.x + cards[0].bbox.width > 236


def test_normalize_graph_ignores_non_icon_shape_between_info_cards():
    components = [
        Component(id="strength-icon-bg", type="shape", bbox=BBox(x=40, y=90, width=36, height=62), source="opencv-residual"),
        Component(id="strength-icon", type="icon", bbox=BBox(x=40, y=90, width=36, height=62), source="sam3"),
        Component(id="strength-title", type="text", bbox=BBox(x=98, y=94, width=58, height=24), text="강도", source="paddleocr"),
        Component(id="strength-desc", type="text", bbox=BBox(x=92, y=128, width=124, height=18), text="최고수온/최저염분", source="paddleocr"),
        Component(id="freq-icon-bg", type="shape", bbox=BBox(x=260, y=90, width=54, height=58), source="opencv-residual"),
        Component(id="freq-icon", type="icon", bbox=BBox(x=260, y=90, width=54, height=58), source="sam3"),
        Component(id="freq-title", type="text", bbox=BBox(x=330, y=94, width=58, height=24), text="빈도", source="paddleocr"),
        Component(id="freq-desc", type="text", bbox=BBox(x=318, y=128, width=100, height=18), text="고수온·저염", source="paddleocr"),
        Component(id="chart-fragment", type="shape", bbox=BBox(x=278, y=40, width=30, height=64), source="opencv-residual"),
        Component(id="duration-icon-bg", type="shape", bbox=BBox(x=520, y=90, width=54, height=58), source="opencv-residual"),
        Component(id="duration-icon", type="icon", bbox=BBox(x=520, y=90, width=54, height=58), source="sam3"),
        Component(id="duration-title", type="text", bbox=BBox(x=590, y=94, width=82, height=24), text="지속시간", source="paddleocr"),
        Component(id="duration-desc", type="text", bbox=BBox(x=582, y=128, width=100, height=18), text="지속일수", source="paddleocr"),
    ]

    normalized = normalize_component_graph(components)

    cards = [component for component in normalized if component.source == "synthetic-info-card-shape"]
    assert len(cards) == 3
    assert cards[0].bbox.x + cards[0].bbox.width < cards[1].bbox.x
    assert cards[1].bbox.x + cards[1].bbox.width < cards[2].bbox.x
    assert any(_containment_ratio(BBox(x=260, y=90, width=54, height=58), card.bbox) >= 0.75 for card in cards)
    assert not any(_containment_ratio(BBox(x=278, y=40, width=30, height=64), card.bbox) >= 0.75 for card in cards)


def test_normalize_graph_does_not_synthesize_info_card_from_long_message_bar():
    components = [
        Component(id="icon-bg", type="shape", bbox=BBox(x=106, y=209, width=65, height=38), source="opencv-residual"),
        Component(id="icon", type="icon", bbox=BBox(x=109, y=210, width=61, height=37), source="sam3"),
        Component(id="label", type="text", bbox=BBox(x=191, y=210, width=126, height=32), text="핵심 메시지", source="paddleocr"),
        Component(
            id="long-message",
            type="text",
            bbox=BBox(x=363, y=211, width=1102, height=33),
            text="평균값만으로는 설명되지 않는 생산량 급감은 고수온·저염·빈영양 등 극한 이벤트로 더 잘 설명할 수 있습니다.",
            source="paddleocr",
        ),
    ]

    normalized = normalize_component_graph(components)

    assert not [component for component in normalized if component.source == "synthetic-info-card-shape"]


def test_normalize_graph_keeps_card_children_but_hides_opencv_fragments():
    components = [
        Component(id="card", type="shape", bbox=BBox(x=20, y=20, width=180, height=70), source="opencv-residual"),
        Component(id="shadow", type="shape", bbox=BBox(x=25, y=82, width=160, height=4), source="opencv-residual"),
        Component(id="icon", type="icon", bbox=BBox(x=42, y=36, width=32, height=32), source="sam3"),
        Component(id="label", type="text", bbox=BBox(x=86, y=42, width=80, height=16), text="label", source="paddleocr"),
    ]

    normalized = normalize_component_graph(components)

    by_id = {component.id: component for component in normalized}
    assert by_id["card"].hidden is False
    assert by_id["shadow"].hidden is True
    assert by_id["icon"].hidden is False
    assert by_id["label"].hidden is False


def test_normalize_graph_hides_text_duplicate_visual_fragment():
    components = [
        Component(
            id="ocr",
            type="text",
            bbox=BBox(x=40, y=40, width=100, height=16),
            text="핵심 메시지",
            source="paddleocr",
        ),
        Component(id="text-fragment", type="icon", bbox=BBox(x=38, y=38, width=106, height=20), source="opencv-residual"),
        Component(id="real-icon", type="icon", bbox=BBox(x=150, y=40, width=28, height=28), source="sam3"),
    ]

    normalized = normalize_component_graph(components)

    by_id = {component.id: component for component in normalized}
    assert by_id["ocr"].hidden is False
    assert by_id["text-fragment"].hidden is True
    assert by_id["real-icon"].hidden is False


def test_normalize_graph_hides_thin_opencv_border_fragments():
    components = [
        Component(id="chart", type="chart", bbox=BBox(x=20, y=20, width=180, height=90), source="sam3"),
        Component(id="left-border", type="icon", bbox=BBox(x=18, y=20, width=3, height=90), source="opencv-residual"),
        Component(id="bottom-border", type="shape", bbox=BBox(x=20, y=112, width=180, height=3), source="opencv-residual"),
        Component(id="real-icon", type="icon", bbox=BBox(x=220, y=32, width=34, height=34), source="sam3"),
    ]

    normalized = normalize_component_graph(components)

    by_id = {component.id: component for component in normalized}
    assert by_id["left-border"].hidden is True
    assert by_id["bottom-border"].hidden is True
    assert by_id["real-icon"].hidden is False


def test_residual_shape_keeps_full_background_under_claimed_image(tmp_path):
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (180, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 160, 70], fill="#dbeafe")
    draw.ellipse([70, 28, 110, 68], fill="#d71920")
    image.save(image_path)

    store = ProjectStore(Settings(data_dir=tmp_path / "data"))
    claimed_image = Component(
        id="photo",
        type="image",
        bbox=BBox(x=70, y=28, width=40, height=40),
        source="sam3",
    )

    components = Analyzer()._residual_visual_components(
        image_path,
        store,
        "project-test",
        [claimed_image],
    )

    shape = next(
        component
        for component in components
        if component.type == "shape"
        and component.bbox.x <= 22
        and component.bbox.y <= 22
        and component.bbox.x + component.bbox.width >= 158
        and component.bbox.y + component.bbox.height >= 68
    )
    assert shape.asset_path


def test_residual_shape_keeps_colored_label_behind_ocr_text(tmp_path):
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (220, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([36, 24, 168, 54], fill="#132a4d")
    draw.rectangle([62, 32, 138, 45], fill="white")
    image.save(image_path)

    store = ProjectStore(Settings(data_dir=tmp_path / "data"))
    text = Component(
        id="label-text",
        type="text",
        bbox=BBox(x=58, y=28, width=86, height=22),
        text="핵심 메시지",
        source="paddleocr",
    )

    components = Analyzer()._residual_visual_components(
        image_path,
        store,
        "project-test",
        [text],
    )

    label_shape = next(
        (
            component
            for component in components
            if component.type == "shape"
            and component.bbox.x <= 40
            and component.bbox.y <= 28
            and component.bbox.x + component.bbox.width >= 164
            and component.bbox.y + component.bbox.height >= 50
        ),
        None,
    )
    assert label_shape is not None


def test_analysis_records_fallback_reasons_when_ai_runtime_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PPT_AGENT_DISABLE_PADDLEOCR", "1")
    monkeypatch.setenv("PPT_AGENT_DISABLE_SAM3", "1")
    project, store = image_project(tmp_path)
    analyzed = Analyzer().analyze(project, store)

    assert analyzed.analysis_notes
    assert any("PaddleOCR" in note for note in analyzed.analysis_notes)
    assert any("SAM3" in note for note in analyzed.analysis_notes)
