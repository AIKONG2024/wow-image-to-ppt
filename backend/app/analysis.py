from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .models import BBox, Component, Project
from .reconstruction import layered_shape_picture_split
from .storage import ProjectStore

SAM_PROMPTS = ["icon", "chart", "graph", "diagram", "table", "arrow", "logo", "badge", "photo", "illustration"]
HF_TOKEN_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def runtime_status() -> dict[str, Any]:
    _ensure_local_ai_cache()
    torch_status: dict[str, Any] = {"available": importlib.util.find_spec("torch") is not None, "cuda_available": False}
    status: dict[str, Any] = {
        "paddleocr": _module_report("paddleocr"),
        "paddle": _module_report("paddle", ("paddlepaddle-gpu", "paddlepaddle")),
        "sam3": _module_report("sam3"),
        "torch": torch_status,
        "hf": {"token_present": _hf_token_present()},
    }
    ocr_worker = _ocr_worker_status()
    if ocr_worker:
        status["paddleocr"]["python"] = ocr_worker.get("python")
        status["paddleocr"]["worker_available"] = ocr_worker.get("paddleocr", False)
        status["paddle"]["worker_available"] = ocr_worker.get("paddle", False)
        status["paddleocr"]["available"] = bool(status["paddleocr"]["available"] or ocr_worker.get("paddleocr"))
        status["paddle"]["available"] = bool(status["paddle"]["available"] or ocr_worker.get("paddle"))
    try:
        import torch

        torch_status.update(
            {
                "version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:  # pragma: no cover - depends on local runtime
        torch_status["error"] = str(exc)

    status["paddleocr"]["ready"] = bool(status["paddleocr"]["available"] and status["paddle"]["available"])
    status["paddleocr"]["mode"] = "subprocess"
    status["sam3"]["ready"] = bool(status["sam3"]["available"] and torch_status["cuda_available"])
    status["sam3"]["hf_token_present"] = status["hf"]["token_present"]

    issues = []
    if not status["paddleocr"]["available"]:
        issues.append("PaddleOCR package is missing; text boxes will not be OCR-based.")
    elif not status["paddle"]["available"]:
        issues.append("PaddleOCR is installed but PaddlePaddle is missing.")
    if not status["sam3"]["available"]:
        issues.append("SAM3 package is missing; visual components use OpenCV fallback.")
    elif not torch_status["cuda_available"]:
        issues.append("SAM3 is installed but CUDA PyTorch is not active.")
    if status["sam3"]["available"] and not status["hf"]["token_present"]:
        issues.append("Hugging Face token is not set; checkpoint download may fail unless it is already cached.")

    status["issues"] = issues
    status["analysis_mode"] = "sam3+paddleocr" if status["sam3"]["ready"] and status["paddleocr"]["ready"] else "opencv-fallback"
    return status


class Analyzer:
    def analyze(self, project: Project, store: ProjectStore) -> Project:
        image_path = Path(project.image_path)
        notes: list[str] = []
        components: list[Component] = []
        text_components = self._paddle_ocr_components(image_path, notes)
        components.extend(text_components)
        sam_components = self._sam3_components(image_path, store, project.id, text_components, notes)
        components.extend(sam_components)
        if not sam_components:
            notes.append("SAM3 result is unavailable; visual components were detected with OpenCV fallback.")
            components.extend(self._opencv_visual_components(image_path, components))
        components.extend(self._residual_visual_components(image_path, store, project.id, components))

        components = _split_layered_visual_components(image_path, components)
        components = _dedupe_components(components)
        project.components = normalize_component_graph(components)
        _refine_chart_component_bboxes(image_path, project.components)
        project.status = "analyzed"
        project.analysis_notes = notes
        project.error = None
        return project

    def _paddle_ocr_components(self, image_path: Path, notes: list[str]) -> list[Component]:
        if os.getenv("PPT_AGENT_DISABLE_PADDLEOCR") == "1":
            notes.append("PaddleOCR is disabled by PPT_AGENT_DISABLE_PADDLEOCR.")
            return []
        if os.getenv("PPT_AGENT_PADDLEOCR_IN_PROCESS") != "1":
            return _paddle_ocr_components_subprocess(image_path, notes)
        if importlib.util.find_spec("paddleocr") is None:
            notes.append("PaddleOCR package is missing; editable text detection was skipped.")
            return []
        try:
            ocr = _create_paddle_ocr()
            result = _run_paddle_ocr(ocr, image_path)
        except Exception as exc:
            notes.append(f"PaddleOCR failed: {_short_error(exc)}")
            return []

        components = _paddle_results_to_components(result)
        if not components:
            notes.append("PaddleOCR ran but returned no readable text boxes.")
        return components

    def _sam3_components(
        self,
        image_path: Path,
        store: ProjectStore,
        project_id: str,
        text_components: list[Component],
        notes: list[str],
    ) -> list[Component]:
        if os.getenv("PPT_AGENT_DISABLE_SAM3") == "1":
            notes.append("SAM3 is disabled by PPT_AGENT_DISABLE_SAM3.")
            return []
        if importlib.util.find_spec("sam3") is None:
            notes.append("SAM3 package is missing; semantic visual segmentation was skipped.")
            return []
        try:
            _ensure_local_ai_cache()
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            image = Image.open(image_path).convert("RGB")
            sam3_context = _sam3_cuda_context_factory(notes)
            with sam3_context():
                model = build_sam3_image_model()
                processor = Sam3Processor(model)
                state = processor.set_image(image)
        except Exception as exc:
            notes.append(f"SAM3 initialization failed: {_short_error(exc)}")
            return []

        components: list[Component] = []
        for prompt in SAM_PROMPTS:
            try:
                with sam3_context():
                    output = processor.set_text_prompt(state=state, prompt=prompt)
            except Exception as exc:
                notes.append(f"SAM3 prompt '{prompt}' failed: {_short_error(exc)}")
                continue
            boxes = _to_box_list(output.get("boxes"))
            masks = output.get("masks")
            scores = output.get("scores")
            for index, box in enumerate(_to_box_list(boxes)):
                score = _float_at(scores, index)
                if score is not None and score < float(os.getenv("PPT_AGENT_SAM3_SCORE_THRESHOLD", "0.25")):
                    continue
                mask_path = None
                mask = _item_at(masks, index)
                if mask is not None:
                    mask_path = _save_mask_asset(store, project_id, mask, index, prompt)
                components.append(
                    Component(
                        id=f"component-{uuid.uuid4().hex[:10]}",
                        type=_prompt_to_type(prompt),
                        bbox=box,
                        mask_path=mask_path,
                        confidence=score,
                        source="sam3",
                    )
                )
        if not components:
            notes.append("SAM3 ran but returned no accepted visual components.")
        return components

    def _opencv_visual_components(self, image_path: Path, text_components: list[Component]) -> list[Component]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        components: list[Component] = []
        image_area = image.shape[0] * image.shape[1]
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area < max(900, image_area * 0.003):
                continue
            bbox = BBox(x=float(x), y=float(y), width=float(width), height=float(height))
            components.append(
                Component(
                    id=f"component-{uuid.uuid4().hex[:10]}",
                    type=_classify_visual(width, height, image.shape[1], image.shape[0]),
                    bbox=bbox,
                    confidence=None,
                    source="opencv-fallback",
                )
            )
        return components

    def _residual_visual_components(
        self,
        image_path: Path,
        store: ProjectStore,
        project_id: str,
        claimed_components: list[Component],
    ) -> list[Component]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return []

        residual_mask = _foreground_mask(image)
        text_components = [component for component in claimed_components if component.type == "text"]
        text_mask = np.zeros(residual_mask.shape, dtype=np.uint8)
        for component in claimed_components:
            if component.type == "text":
                _paint_claimed_component(text_mask, component, residual_mask.shape)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        residual_mask = cv2.morphologyEx(residual_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        residual_mask = cv2.morphologyEx(residual_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(residual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = image.shape[0] * image.shape[1]
        components: list[Component] = []
        for index, contour in enumerate(contours):
            x, y, width, height = cv2.boundingRect(contour)
            pixel_area = int(cv2.contourArea(contour))
            bbox_area = width * height
            if pixel_area < max(50, image_area * 0.00012) and bbox_area < max(400, image_area * 0.0007):
                continue
            if bbox_area > image_area * 0.82:
                continue
            bbox = BBox(x=float(x), y=float(y), width=float(width), height=float(height))

            component_mask = np.zeros_like(residual_mask)
            cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
            if _is_text_only_residual(component_mask, residual_mask, text_mask):
                continue
            asset_path = _save_masked_crop_asset(
                store,
                project_id,
                image,
                component_mask,
                bbox,
                f"residual-{index}",
            )
            components.append(
                Component(
                    id=f"component-{uuid.uuid4().hex[:10]}",
                    type=_classify_visual(width, height, image.shape[1], image.shape[0]),
                    bbox=bbox,
                    asset_path=asset_path,
                    confidence=None,
                    source="opencv-residual",
                )
            )
        return components


def _prompt_to_type(prompt: str) -> str:
    if prompt == "graph":
        return "chart"
    if prompt == "illustration":
        return "image"
    mapping = {"photo": "image", "logo": "icon", "badge": "icon"}
    return mapping.get(prompt, prompt if prompt in {"icon", "chart", "diagram", "table", "arrow"} else "unknown")


def _classify_visual(width: int, height: int, image_width: int, image_height: int) -> str:
    if height < image_height * 0.09 or (width > image_width * 0.55 and height < image_height * 0.16):
        return "shape"
    if width > image_width * 0.45 and height < image_height * 0.55:
        return "shape"
    if width > image_width * 0.25 and height > image_height * 0.18:
        return "chart"
    if min(width, height) < max(image_width, image_height) * 0.14:
        return "icon"
    return "shape"


def _sam3_cuda_context_factory(notes: list[str]):
    def no_context():
        return nullcontext()

    if importlib.util.find_spec("torch") is None:
        return no_context
    try:
        import torch
    except Exception as exc:
        notes.append(f"SAM3 could not import PyTorch: {_short_error(exc)}")
        return no_context
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        notes.append(f"SAM3 could not check CUDA availability: {_short_error(exc)}")
        return no_context
    if not cuda_available:
        notes.append("SAM3 is installed, but this Python environment is using CPU PyTorch.")
        return no_context

    _enable_torch_tf32(torch)
    autocast = getattr(torch, "autocast", None)
    dtype = getattr(torch, "bfloat16", None)
    if autocast is None or dtype is None:
        notes.append("SAM3 CUDA autocast is unavailable; running without autocast.")
        return no_context

    def cuda_context():
        return autocast("cuda", dtype=dtype)

    return cuda_context


def _enable_torch_tf32(torch: Any) -> None:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _overlap_ratio(left: BBox, right: BBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min(left.width * left.height, right.width * right.height)
    return intersection / smaller if smaller else 0.0


def _dedupe_components(components: list[Component]) -> list[Component]:
    ordered = sorted(components, key=lambda item: (item.bbox.y, item.bbox.x))
    kept: list[Component] = []
    for component in ordered:
        if any(_is_duplicate_component(component, existing) for existing in kept):
            continue
        kept.append(component)
    return kept


def _split_layered_visual_components(image_path: Path, components: list[Component]) -> list[Component]:
    text_components = [component for component in components if component.type == "text"]
    try:
        source = Image.open(image_path).convert("RGBA")
    except Exception:
        return components

    split_components: list[Component] = []
    with source:
        for component in components:
            if component.type == "shape" and _has_contained_visual_child(component, components):
                split_components.append(component)
                continue
            split = layered_shape_picture_split(component, source, text_components)
            if split is None:
                split_components.append(component)
                continue
            _, _, _, inner_bbox = split
            split_components.extend(
                [
                    Component(
                        id=f"component-{uuid.uuid4().hex[:10]}",
                        type="shape",
                        bbox=component.bbox,
                        confidence=component.confidence,
                        source=f"{component.source}-layer-shape",
                    ),
                    Component(
                        id=f"component-{uuid.uuid4().hex[:10]}",
                        type=component.type if component.type in {"icon", "image"} else "image",
                        bbox=inner_bbox,
                        confidence=component.confidence,
                        source=f"{component.source}-layer-image",
                    ),
                ]
            )
    return split_components


def normalize_component_graph(components: list[Component]) -> list[Component]:
    normalized = [
        component.model_copy(deep=True)
        for component in components
        if not component.source.startswith("synthetic-")
    ]
    _hide_oversized_chart_parents(normalized)
    _synthesize_chart_panel_shapes(normalized)
    _expand_charts_to_nearby_annotations(normalized)
    _hide_chart_children(normalized)
    _hide_shape_fragments(normalized)
    _hide_text_duplicate_visual_fragments(normalized)
    _hide_thin_opencv_rule_fragments(normalized)
    _synthesize_frame_shapes(normalized)
    _synthesize_info_card_shapes(normalized)
    return normalized


def _refine_chart_component_bboxes(image_path: Path, components: list[Component]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for component in components:
        if component.hidden or component.type not in {"chart", "table"}:
            continue
        if _bbox_area(component.bbox) < 1200:
            continue
        refined = _chart_content_bbox(rgb, component, components)
        if refined is None:
            continue
        if _bbox_changed(component.bbox, refined):
            component.bbox = refined
    _restore_chart_overlay_labels(components)
    _hide_chart_internal_text_labels(components)


def _chart_content_bbox(rgb: np.ndarray, component: Component, components: list[Component]) -> BBox | None:
    image_height, image_width = rgb.shape[:2]
    bbox = _chart_search_bbox(component, components)
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(image_width, int(round(bbox.x + bbox.width)))
    y2 = min(image_height, int(round(bbox.y + bbox.height)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = rgb[y1:y2, x1:x2]
    values = crop.astype(np.int16)
    distance_from_white = np.max(np.abs(values - 255), axis=2)
    channel_spread = values.max(axis=2) - values.min(axis=2)
    mask = (distance_from_white > 16) | (channel_spread > 12)
    _remove_chart_edge_rules(mask)
    for other in components:
        if other.id == component.id:
            continue
        if other.type == "text" and _is_chart_title_overlay(bbox, other.bbox):
            _clear_local_bbox(mask, bbox, other.bbox, padding=3)
            continue
        if other.type == "shape" and _is_chart_title_overlay(bbox, other.bbox):
            _clear_local_bbox(mask, bbox, other.bbox, padding=4)
    if int(np.count_nonzero(mask)) < 24:
        return None
    rows, columns = np.where(mask)
    local_x1 = int(columns.min())
    local_x2 = int(columns.max()) + 1
    local_y1 = int(rows.min())
    local_y2 = int(rows.max()) + 1
    pad_x = max(4, int(round((x2 - x1) * 0.012)))
    pad_y = max(4, int(round((y2 - y1) * 0.02)))
    refined = BBox(
        x=float(max(x1, x1 + local_x1 - pad_x)),
        y=float(max(y1, y1 + local_y1 - pad_y)),
        width=float(min(x2, x1 + local_x2 + pad_x) - max(x1, x1 + local_x1 - pad_x)),
        height=float(min(y2, y1 + local_y2 + pad_y) - max(y1, y1 + local_y1 - pad_y)),
    )
    if refined.width < bbox.width * 0.18 or refined.height < bbox.height * 0.18:
        return None
    return refined


def _chart_search_bbox(component: Component, components: list[Component]) -> BBox:
    frame_candidates = [
        candidate
        for candidate in components
        if not candidate.hidden
        and candidate.type == "shape"
        and candidate.source == "synthetic-frame-shape"
        and _intersection_area(component.bbox, candidate.bbox) >= _bbox_area(component.bbox) * 0.45
        and _bbox_area(candidate.bbox) <= max(_bbox_area(component.bbox) * 4.0, _bbox_area(component.bbox) + 4000)
    ]
    if frame_candidates:
        return min(frame_candidates, key=lambda candidate: _bbox_area(candidate.bbox)).bbox

    panel_candidates = [
        candidate
        for candidate in components
        if not candidate.hidden
        and candidate.type == "shape"
        and candidate.source == "synthetic-chart-panel-shape"
        and _intersection_area(component.bbox, candidate.bbox) >= _bbox_area(component.bbox) * 0.45
        and _bbox_area(candidate.bbox) <= max(_bbox_area(component.bbox) * 5.0, _bbox_area(component.bbox) + 12000)
    ]
    if panel_candidates:
        panel = min(panel_candidates, key=lambda candidate: _bbox_area(candidate.bbox)).bbox
        return _chart_panel_search_bbox(panel, components)
    return component.bbox


def _chart_panel_search_bbox(panel: BBox, components: list[Component]) -> BBox:
    card_tops = [
        component.bbox.y
        for component in components
        if not component.hidden
        and component.source == "synthetic-info-card-shape"
        and _intersection_area(component.bbox, panel) > 0
        and component.bbox.y > panel.y + panel.height * 0.45
    ]
    if not card_tops:
        return panel
    bottom = min(card_tops) - 6.0
    if bottom <= panel.y + panel.height * 0.25:
        return panel
    return BBox(x=panel.x, y=panel.y, width=panel.width, height=bottom - panel.y)


def _restore_chart_overlay_labels(components: list[Component]) -> None:
    charts = [
        component
        for component in components
        if not component.hidden and component.type in {"chart", "table"} and _bbox_area(component.bbox) >= 1200
    ]
    if not charts:
        return
    for candidate in components:
        if not candidate.hidden or candidate.type not in {"text", "shape"}:
            continue
        if candidate.type == "shape" and not candidate.source.startswith("opencv"):
            continue
        if candidate.bbox.width < 36 or candidate.bbox.height > 64:
            continue
        for chart in charts:
            if candidate.bbox.y + candidate.bbox.height > chart.bbox.y + 8:
                continue
            if candidate.bbox.y + candidate.bbox.height < chart.bbox.y - 96:
                continue
            if _horizontal_overlap_ratio(candidate.bbox, chart.bbox) < 0.12:
                continue
            candidate.hidden = False
            break


def _hide_chart_internal_text_labels(components: list[Component]) -> None:
    charts = [
        component
        for component in components
        if not component.hidden and component.type in {"chart", "table"} and _bbox_area(component.bbox) >= 1200
    ]
    for text in components:
        if text.hidden or text.type != "text":
            continue
        for chart in charts:
            if text.bbox.y + text.bbox.height <= chart.bbox.y + 4:
                continue
            if _intersection_area(text.bbox, chart.bbox) < _bbox_area(text.bbox) * 0.25:
                continue
            text.hidden = True
            break


def _bbox_changed(left: BBox, right: BBox) -> bool:
    return (
        abs(left.x - right.x) > 1
        or abs(left.y - right.y) > 1
        or abs(left.width - right.width) > 1
        or abs(left.height - right.height) > 1
    )


def _remove_chart_edge_rules(mask: np.ndarray) -> None:
    height, width = mask.shape
    if height < 12 or width < 12:
        return
    margin = max(3, min(14, int(round(min(width, height) * 0.04))))
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False


def _clear_local_bbox(mask: np.ndarray, container: BBox, target: BBox, padding: int) -> None:
    x1 = max(0, int(np.floor(target.x - container.x)) - padding)
    y1 = max(0, int(np.floor(target.y - container.y)) - padding)
    x2 = min(mask.shape[1], int(np.ceil(target.x + target.width - container.x)) + padding)
    y2 = min(mask.shape[0], int(np.ceil(target.y + target.height - container.y)) + padding)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = False


def _is_chart_title_overlay(chart_bbox: BBox, candidate: BBox) -> bool:
    if not _intersects(chart_bbox, candidate):
        return False
    if candidate.height > max(64.0, chart_bbox.height * 0.3):
        return False
    if candidate.width < chart_bbox.width * 0.12:
        return False
    candidate_center_y = candidate.y + candidate.height / 2
    return candidate_center_y <= chart_bbox.y + chart_bbox.height * 0.14


def _synthesize_chart_panel_shapes(components: list[Component]) -> None:
    hidden_panels = [
        component
        for component in components
        if component.hidden and component.type in {"chart", "table"} and _bbox_area(component.bbox) >= 1800
    ]
    visible_charts = [
        component
        for component in components
        if not component.hidden and component.type in {"chart", "table"} and _bbox_area(component.bbox) >= 800
    ]
    for panel in hidden_panels:
        if _has_similar_visible_shape(panel.bbox, components):
            continue
        contained_charts = [
            chart
            for chart in visible_charts
            if chart.id != panel.id
            and _bbox_area(chart.bbox) < _bbox_area(panel.bbox) * 0.8
            and _containment_ratio(chart.bbox, panel.bbox) >= 0.65
        ]
        if not contained_charts:
            continue
        components.append(
            Component(
                id=f"component-{uuid.uuid4().hex[:10]}",
                type="shape",
                bbox=panel.bbox,
                source="synthetic-chart-panel-shape",
            )
        )


def _synthesize_frame_shapes(components: list[Component]) -> None:
    border_fragments = [
        component
        for component in components
        if component.source.startswith("opencv")
        and component.type in {"icon", "shape", "unknown", "line"}
        and not component.source.endswith("-layer-shape")
    ]
    verticals = [component for component in border_fragments if _is_thin_vertical_bbox(component.bbox)]
    horizontals = [component for component in border_fragments if _is_thin_horizontal_bbox(component.bbox)]
    verticals.sort(key=lambda component: (component.bbox.x, component.bbox.y))
    consumed: set[str] = set()
    for left in verticals:
        if left.id in consumed:
            continue
        candidates = [
            right
            for right in verticals
            if right.id not in consumed
            and right.id != left.id
            and right.bbox.x > left.bbox.x + 24
            and _vertical_overlap_ratio(left.bbox, right.bbox) >= 0.65
        ]
        if not candidates:
            continue
        right = min(
            candidates,
            key=lambda item: (
                abs((left.bbox.y + left.bbox.height / 2) - (item.bbox.y + item.bbox.height / 2)),
                item.bbox.x - left.bbox.x,
            ),
        )
        bbox = _bbox_union([left.bbox, right.bbox])
        edge_fragments = _matching_frame_horizontal_fragments(bbox, horizontals)
        if edge_fragments:
            bbox = _bbox_union([bbox, *[edge.bbox for edge in edge_fragments]])
        if bbox.width < 32 or bbox.height < 28:
            continue
        if _has_similar_visible_shape(bbox, components):
            continue
        components.append(
            Component(
                id=f"component-{uuid.uuid4().hex[:10]}",
                type="shape",
                bbox=bbox,
                source="synthetic-frame-shape",
            )
        )
        consumed.add(left.id)
        consumed.add(right.id)


def _synthesize_info_card_shapes(components: list[Component]) -> None:
    visible_icons = [
        component
        for component in components
        if not component.hidden and component.type == "icon" and component.source.startswith("sam3")
    ]
    icon_backgrounds = [
        component
        for component in components
        if component.type == "shape"
        and component.source.startswith("opencv")
        and 22 <= component.bbox.width <= 96
        and 28 <= component.bbox.height <= 120
        and _bbox_area(component.bbox) <= 9000
        and _has_overlapping_visible_icon(component.bbox, visible_icons)
    ]
    icon_backgrounds.sort(key=lambda component: component.bbox.x)
    texts = [component for component in components if not component.hidden and component.type == "text" and component.text]
    for index, icon_bg in enumerate(icon_backgrounds):
        if _has_larger_card_shape(icon_bg, components):
            continue
        next_icon = icon_backgrounds[index + 1] if index + 1 < len(icon_backgrounds) else None
        group_texts = _nearby_info_card_texts(icon_bg.bbox, next_icon.bbox if next_icon else None, texts)
        if len(group_texts) < 2:
            continue
        union = _bbox_union([icon_bg.bbox, *[text.bbox for text in group_texts]])
        left_pad = _info_card_left_padding(icon_bg.bbox)
        right_pad = max(8.0, icon_bg.bbox.width * 0.18)
        vertical_pad = max(8.0, icon_bg.bbox.height * 0.18)
        bbox = BBox(
            x=union.x - left_pad,
            y=union.y - vertical_pad,
            width=union.width + left_pad + right_pad,
            height=union.height + vertical_pad * 2,
        )
        if next_icon is not None:
            max_right = next_icon.bbox.x - max(8.0, icon_bg.bbox.width * 0.2)
            if bbox.x + bbox.width > max_right:
                bbox = BBox(x=bbox.x, y=bbox.y, width=max(1.0, max_right - bbox.x), height=bbox.height)
        if bbox.width < icon_bg.bbox.width * 1.6 or _has_similar_visible_shape(bbox, components):
            continue
        components.append(
            Component(
                id=f"component-{uuid.uuid4().hex[:10]}",
                type="shape",
                bbox=bbox,
                source="synthetic-info-card-shape",
            )
        )


def _info_card_left_padding(icon_bbox: BBox) -> float:
    narrow_icon_extra = max(0.0, icon_bbox.height - icon_bbox.width) + 2.0
    return max(8.0, icon_bbox.width * 0.28, narrow_icon_extra)


def _matching_frame_horizontal_fragments(bbox: BBox, horizontals: list[Component]) -> list[Component]:
    edges: list[Component] = []
    for horizontal in horizontals:
        center_y = horizontal.bbox.y + horizontal.bbox.height / 2
        if _horizontal_overlap_ratio(horizontal.bbox, bbox) < 0.65:
            continue
        if abs(center_y - bbox.y) <= 10 or abs(center_y - (bbox.y + bbox.height)) <= 10:
            edges.append(horizontal)
    return edges[:2]


def _nearby_info_card_texts(icon_bbox: BBox, next_icon_bbox: BBox | None, texts: list[Component]) -> list[Component]:
    lower = icon_bbox.y - max(10.0, icon_bbox.height * 0.2)
    upper = icon_bbox.y + icon_bbox.height + max(34.0, icon_bbox.height * 0.6)
    left = icon_bbox.x + icon_bbox.width * 0.55
    right = icon_bbox.x + 260.0
    if next_icon_bbox is not None:
        right = min(right, next_icon_bbox.x + next_icon_bbox.width + 12)
    result: list[Component] = []
    for text in texts:
        if text.bbox.y + text.bbox.height < lower or text.bbox.y > upper:
            continue
        if text.bbox.x < left or text.bbox.x > right:
            continue
        if text.bbox.x + text.bbox.width > right + 24:
            continue
        if text.bbox.width > 240:
            continue
        result.append(text)
    return result


def _has_overlapping_visible_icon(bbox: BBox, icons: list[Component]) -> bool:
    for icon in icons:
        if _containment_ratio(icon.bbox, bbox) >= 0.55:
            return True
        if _containment_ratio(bbox, icon.bbox) >= 0.55:
            return True
    return False


def _has_larger_card_shape(icon_bg: Component, components: list[Component]) -> bool:
    for component in components:
        if component.id == icon_bg.id or component.hidden or component.type != "shape":
            continue
        if component.source in {"synthetic-chart-panel-shape", "synthetic-frame-shape"}:
            continue
        if _bbox_area(component.bbox) <= _bbox_area(icon_bg.bbox) * 1.8:
            continue
        if _bbox_area(component.bbox) > _bbox_area(icon_bg.bbox) * 18:
            continue
        if _containment_ratio(icon_bg.bbox, component.bbox) >= 0.75:
            return True
    return False


def _has_similar_visible_shape(bbox: BBox, components: list[Component]) -> bool:
    for component in components:
        if component.hidden or component.type != "shape":
            continue
        if _bbox_area_similarity(component.bbox, bbox) < 0.72:
            continue
        if _overlap_ratio(component.bbox, bbox) > 0.85:
            return True
    return False


def _is_thin_vertical_bbox(bbox: BBox) -> bool:
    return bbox.width <= 6 and bbox.height >= 28


def _is_thin_horizontal_bbox(bbox: BBox) -> bool:
    return bbox.height <= 6 and bbox.width >= 28


def _vertical_overlap_ratio(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    return overlap / max(1.0, min(left.height, right.height))


def _hide_oversized_chart_parents(components: list[Component]) -> None:
    charts = [
        component
        for component in components
        if not component.hidden and component.type in {"chart", "table"} and _bbox_area(component.bbox) >= 800
    ]
    for parent in charts:
        if parent.hidden:
            continue
        inner_charts = [
            child
            for child in charts
            if child.id != parent.id
            and not child.hidden
            and _bbox_area(child.bbox) < _bbox_area(parent.bbox) * 0.55
            and _containment_ratio(child.bbox, parent.bbox) >= 0.82
        ]
        if not inner_charts:
            continue
        best_child = max(inner_charts, key=lambda child: _bbox_area(child.bbox))
        if _has_non_chart_content_below(parent, best_child, components):
            parent.hidden = True


def _expand_charts_to_nearby_annotations(components: list[Component]) -> None:
    charts = [
        component
        for component in components
        if not component.hidden and component.type in {"chart", "table"} and _bbox_area(component.bbox) >= 800
    ]
    hidden_parents = [
        component
        for component in components
        if component.hidden and component.type in {"chart", "table"} and _bbox_area(component.bbox) >= 800
    ]
    for chart in charts:
        parent = _smallest_hidden_parent(chart, hidden_parents)
        if parent is None:
            continue
        candidate_region = parent.bbox
        lower_limit = _chart_lower_annotation_limit(chart, candidate_region, components)
        boxes = [chart.bbox]
        for candidate in components:
            if candidate.id == chart.id or candidate.hidden:
                continue
            if candidate.type not in {"text", "shape", "line", "unknown"}:
                continue
            if _containment_ratio(candidate.bbox, candidate_region) < 0.55:
                continue
            if candidate.bbox.y + candidate.bbox.height > lower_limit:
                continue
            if not _near_chart_annotation_band(candidate.bbox, chart.bbox):
                continue
            boxes.append(candidate.bbox)
        chart.bbox = _bbox_union(boxes)


def _smallest_hidden_parent(component: Component, parents: list[Component]) -> Component | None:
    containing = [
        parent
        for parent in parents
        if _bbox_area(parent.bbox) > _bbox_area(component.bbox)
        and _containment_ratio(component.bbox, parent.bbox) >= 0.82
    ]
    if not containing:
        return None
    return min(containing, key=lambda parent: _bbox_area(parent.bbox))


def _chart_lower_annotation_limit(chart: Component, region: BBox, components: list[Component]) -> float:
    chart_bottom = chart.bbox.y + chart.bbox.height
    candidates = [
        component.bbox.y
        for component in components
        if component.id != chart.id
        and not component.hidden
        and component.type in {"icon", "image"}
        and component.source.startswith("sam3")
        and component.bbox.y > chart_bottom + 14
        and _containment_ratio(component.bbox, region) >= 0.55
        and _horizontal_overlap_ratio(component.bbox, chart.bbox) > 0.0
    ]
    if candidates:
        return min(candidates) - 4
    return min(region.y + region.height, chart_bottom + 96)


def _near_chart_annotation_band(candidate: BBox, chart: BBox) -> bool:
    return _intersects(candidate, _expanded_bbox(chart, 90, 96))


def _has_non_chart_content_below(parent: Component, chart: Component, components: list[Component]) -> bool:
    chart_bottom = chart.bbox.y + chart.bbox.height
    evidence = 0
    for component in components:
        if component.id in {parent.id, chart.id} or component.hidden:
            continue
        if _containment_ratio(component.bbox, parent.bbox) < 0.65:
            continue
        if component.bbox.y <= chart_bottom + 6:
            continue
        if component.type in {"text", "icon", "image", "shape"}:
            evidence += 1
    return evidence >= 2


def _hide_chart_children(components: list[Component]) -> None:
    chart_parents = [
        component
        for component in components
        if not component.hidden and component.type in {"chart", "table"} and _bbox_area(component.bbox) >= 800
    ]
    chart_parents.sort(key=lambda component: _bbox_area(component.bbox), reverse=True)
    for parent in chart_parents:
        for child in components:
            if child.id == parent.id or child.hidden:
                continue
            if _containment_ratio(child.bbox, parent.bbox) < 0.68:
                continue
            if child.type in {"chart", "table"} and _bbox_area(child.bbox) > _bbox_area(parent.bbox) * 0.72:
                continue
            child.hidden = True


def _hide_shape_fragments(components: list[Component]) -> None:
    shape_parents = [
        component
        for component in components
        if not component.hidden and component.type == "shape" and _bbox_area(component.bbox) >= 1000
    ]
    shape_parents.sort(key=lambda component: _bbox_area(component.bbox), reverse=True)
    for parent in shape_parents:
        for child in components:
            if child.id == parent.id or child.hidden:
                continue
            if not child.source.startswith("opencv"):
                continue
            if child.type not in {"shape", "line", "unknown"}:
                continue
            if _bbox_area(child.bbox) > _bbox_area(parent.bbox) * 0.35:
                continue
            if _containment_ratio(child.bbox, parent.bbox) >= 0.75:
                child.hidden = True


def _hide_text_duplicate_visual_fragments(components: list[Component]) -> None:
    text_components = [component for component in components if not component.hidden and component.type == "text"]
    if not text_components:
        return
    for component in components:
        if component.hidden or component.type == "text":
            continue
        if not component.source.startswith("opencv"):
            continue
        if component.type not in {"icon", "shape", "unknown", "image"}:
            continue
        component_area = _bbox_area(component.bbox)
        if component_area > 9000:
            continue
        text_overlap = sum(_intersection_area(component.bbox, text.bbox) for text in text_components)
        if text_overlap / max(1.0, component_area) >= 0.48:
            component.hidden = True


def _hide_thin_opencv_rule_fragments(components: list[Component]) -> None:
    for component in components:
        if component.hidden or not component.source.startswith("opencv"):
            continue
        if component.type not in {"icon", "shape", "line", "unknown"}:
            continue
        width = component.bbox.width
        height = component.bbox.height
        if (width <= 5 and height >= 28) or (height <= 5 and width >= 28):
            component.hidden = True


def _has_contained_visual_child(component: Component, components: list[Component]) -> bool:
    for candidate in components:
        if candidate.id == component.id or candidate.hidden:
            continue
        if candidate.type not in {"image", "icon", "diagram", "chart"}:
            continue
        if candidate.bbox.width * candidate.bbox.height >= component.bbox.width * component.bbox.height * 0.72:
            continue
        if _containment_ratio(candidate.bbox, component.bbox) >= 0.75:
            return True
    return False


def _containment_ratio(inner: BBox, outer: BBox) -> float:
    x1 = max(inner.x, outer.x)
    y1 = max(inner.y, outer.y)
    x2 = min(inner.x + inner.width, outer.x + outer.width)
    y2 = min(inner.y + inner.height, outer.y + outer.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inner_area = max(1.0, inner.width * inner.height)
    return intersection / inner_area


def _intersection_area(left: BBox, right: BBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersects(left: BBox, right: BBox) -> bool:
    return not (
        left.x + left.width <= right.x
        or right.x + right.width <= left.x
        or left.y + left.height <= right.y
        or right.y + right.height <= left.y
    )


def _horizontal_overlap_ratio(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    shorter = max(1.0, min(left.width, right.width))
    return overlap / shorter


def _expanded_bbox(bbox: BBox, horizontal: float, vertical: float) -> BBox:
    return BBox(
        x=bbox.x - horizontal,
        y=bbox.y - vertical,
        width=bbox.width + horizontal * 2,
        height=bbox.height + vertical * 2,
    )


def _bbox_union(boxes: list[BBox]) -> BBox:
    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.x + box.width for box in boxes)
    y2 = max(box.y + box.height for box in boxes)
    return BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox.width * bbox.height)


def _is_duplicate_component(component: Component, existing: Component) -> bool:
    if component.type == "text" or existing.type == "text":
        return component.type == existing.type and _overlap_ratio(component.bbox, existing.bbox) > 0.85
    if component.type != existing.type or _overlap_ratio(component.bbox, existing.bbox) <= 0.85:
        return False
    return _bbox_area_similarity(component.bbox, existing.bbox) >= 0.72


def _bbox_area_similarity(left: BBox, right: BBox) -> float:
    left_area = max(1.0, left.width * left.height)
    right_area = max(1.0, right.width * right.height)
    return min(left_area, right_area) / max(left_area, right_area)


def _to_box_list(raw_boxes: Any) -> list[BBox]:
    if raw_boxes is None:
        return []
    if isinstance(raw_boxes, list) and all(isinstance(item, BBox) for item in raw_boxes):
        return raw_boxes
    try:
        array = raw_boxes.detach().cpu().numpy() if hasattr(raw_boxes, "detach") else np.asarray(raw_boxes)
    except Exception:
        return []
    boxes: list[BBox] = []
    for item in array:
        values = [float(value) for value in item[:4]]
        x1, y1, x2, y2 = values
        boxes.append(BBox(x=x1, y=y1, width=max(1.0, x2 - x1), height=max(1.0, y2 - y1)))
    return boxes


def _module_report(module_name: str, distributions: tuple[str, ...] | None = None) -> dict[str, Any]:
    available = importlib.util.find_spec(module_name) is not None
    report: dict[str, Any] = {"available": available}
    for distribution in distributions or (module_name,):
        try:
            report["version"] = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    return report


def _ocr_worker_status() -> dict[str, Any] | None:
    python = _paddle_ocr_python()
    if python == sys.executable or not Path(python).exists():
        return None
    try:
        completed = subprocess.run(
            [
                python,
                "-c",
                "import importlib.util, json, sys; "
                "print(json.dumps({'python': sys.executable, 'paddleocr': importlib.util.find_spec('paddleocr') is not None, 'paddle': importlib.util.find_spec('paddle') is not None}))",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return {"python": python, "paddleocr": False, "paddle": False}
    if completed.returncode != 0:
        return {"python": python, "paddleocr": False, "paddle": False}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"python": python, "paddleocr": False, "paddle": False}


def _hf_token_present() -> bool:
    return any(bool(os.getenv(name)) for name in HF_TOKEN_NAMES)


def _create_paddle_ocr():
    _ensure_local_ai_cache()
    from paddleocr import PaddleOCR

    device = os.getenv("PPT_AGENT_PADDLEOCR_DEVICE", "gpu")
    candidates = [
        {
            "lang": "korean",
            "device": device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": "korean",
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": "korean",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {"lang": "korean", "use_angle_cls": True, "show_log": False},
        {"lang": "korean", "use_angle_cls": True},
        {"lang": "korean"},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    return PaddleOCR(lang="korean")


def _run_paddle_ocr(ocr: Any, image_path: Path):
    if hasattr(ocr, "predict"):
        return ocr.predict(input=str(image_path))
    return ocr.ocr(str(image_path), cls=True)


def _paddle_ocr_components_subprocess(image_path: Path, notes: list[str]) -> list[Component]:
    _ensure_local_ai_cache()
    env = os.environ.copy()
    backend_dir = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = backend_dir + os.pathsep + env.get("PYTHONPATH", "")
    timeout = int(os.getenv("PPT_AGENT_PADDLEOCR_TIMEOUT", "300"))
    try:
        completed = subprocess.run(
            [_paddle_ocr_python(), "-m", "app.paddle_ocr_worker", str(image_path)],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        notes.append(f"PaddleOCR subprocess failed: {_short_error(exc)}")
        return []
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        notes.append(f"PaddleOCR subprocess failed: {detail.splitlines()[-1][:240]}")
        return []
    try:
        payload = json.loads(completed.stdout)
        components = [Component.model_validate(item) for item in payload]
    except Exception as exc:
        notes.append(f"PaddleOCR subprocess returned invalid JSON: {_short_error(exc)}")
        return []
    if not components:
        notes.append("PaddleOCR ran but returned no readable text boxes.")
    return components


def _paddle_ocr_python() -> str:
    configured = os.getenv("PPT_AGENT_PADDLEOCR_PYTHON")
    if configured:
        return configured
    project_root = Path(__file__).resolve().parents[2]
    local_python = project_root / ".venv-ocr" / "Scripts" / "python.exe"
    if local_python.exists():
        return str(local_python)
    return sys.executable


def _ensure_local_ai_cache() -> None:
    cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "XDG_CACHE_HOME": cache_dir,
        "HOME": cache_dir / "home",
        "USERPROFILE": cache_dir / "home",
        "PADDLE_HOME": cache_dir / "paddle",
        "PADDLE_PDX_CACHE_HOME": cache_dir / "paddlex",
        "PADDLE_EXTENSION_DIR": cache_dir / "paddle_extension",
        "HF_HOME": cache_dir / "huggingface",
        "PIP_CACHE_DIR": cache_dir / "pip",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def _paddle_results_to_components(result: Any) -> list[Component]:
    components = []
    components.extend(_legacy_paddle_components(result))
    components.extend(_v3_paddle_components(result))
    return _dedupe_components(components)


def _legacy_paddle_components(result: Any) -> list[Component]:
    components: list[Component] = []
    if not isinstance(result, (list, tuple)):
        return components
    for page in result:
        if not isinstance(page, (list, tuple)):
            continue
        for line in page:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            points, payload = line[0], line[1]
            if not isinstance(payload, (list, tuple)) or not payload:
                continue
            text = str(payload[0]).strip()
            confidence = _safe_float(payload[1]) if len(payload) > 1 else None
            bbox = _bbox_from_points(points)
            if text and bbox:
                components.append(_text_component(bbox, text, confidence))
    return components


def _v3_paddle_components(result: Any) -> list[Component]:
    components: list[Component] = []
    for data in _iter_result_dicts(result):
        texts = _as_list(_first_present(data, ("rec_texts", "texts", "text")))
        scores = _as_list(_first_present(data, ("rec_scores", "scores", "text_scores")))
        polygons = _as_list(_first_present(data, ("rec_polys", "dt_polys", "text_polys", "polys")))
        boxes = _as_list(_first_present(data, ("rec_boxes", "dt_boxes", "text_boxes", "boxes")))
        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()
            if not text:
                continue
            bbox = None
            if index < len(polygons):
                bbox = _bbox_from_points(polygons[index])
            if bbox is None and index < len(boxes):
                bbox = _bbox_from_points(boxes[index])
            if bbox is None:
                continue
            confidence = _safe_float(scores[index]) if index < len(scores) else None
            components.append(_text_component(bbox, text, confidence))
    return components


def _iter_result_dicts(result: Any):
    for item in _as_list(result):
        data = _result_dict(item)
        if data is None:
            continue
        if isinstance(data.get("res"), dict):
            data = data["res"]
        yield data


def _result_dict(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item
    for attr in ("res", "json"):
        value = getattr(item, attr, None)
        if callable(value):
            value = value()
        if isinstance(value, dict):
            return value
    return None


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _bbox_from_points(points: Any) -> BBox | None:
    try:
        array = np.asarray(points, dtype=float)
    except Exception:
        return None
    if array.ndim == 1 and array.size >= 4:
        x1, y1, x2, y2 = [float(value) for value in array[:4]]
        return BBox(x=min(x1, x2), y=min(y1, y2), width=abs(x2 - x1), height=abs(y2 - y1))
    if array.ndim >= 2 and array.shape[-1] >= 2:
        flat = array.reshape(-1, array.shape[-1])
        xs = flat[:, 0]
        ys = flat[:, 1]
        return BBox(x=float(xs.min()), y=float(ys.min()), width=float(xs.max() - xs.min()), height=float(ys.max() - ys.min()))
    return None


def _text_component(bbox: BBox, text: str, confidence: float | None) -> Component:
    return Component(
        id=f"component-{uuid.uuid4().hex[:10]}",
        type="text",
        bbox=bbox,
        text=text,
        confidence=confidence,
        source="paddleocr",
    )


def _safe_float(value: Any) -> float | None:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


def _float_at(sequence: Any, index: int) -> float | None:
    item = _item_at(sequence, index)
    return _safe_float(item)


def _item_at(sequence: Any, index: int) -> Any:
    if sequence is None:
        return None
    try:
        return sequence[index]
    except Exception:
        return None


def _short_error(exc: Exception) -> str:
    return str(exc).splitlines()[0][:240] or exc.__class__.__name__


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    non_white = np.logical_or(saturation > 18, value < 246)
    mask = (non_white.astype("uint8") * 255)
    return mask


def _paint_claimed_component(mask: np.ndarray, component: Component, image_shape: tuple[int, int]) -> None:
    if component.mask_path and Path(component.mask_path).exists():
        component_mask = Image.open(component.mask_path).convert("L")
        if component_mask.size != (image_shape[1], image_shape[0]):
            component_mask = component_mask.resize((image_shape[1], image_shape[0]))
        array = np.asarray(component_mask)
        mask[array > 0] = 255
    else:
        padding = _claim_padding(component)
        x1 = max(0, int(np.floor(component.bbox.x - padding)))
        y1 = max(0, int(np.floor(component.bbox.y - padding)))
        x2 = min(image_shape[1], int(np.ceil(component.bbox.x + component.bbox.width + padding)))
        y2 = min(image_shape[0], int(np.ceil(component.bbox.y + component.bbox.height + padding)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255


def _is_text_only_residual(component_mask: np.ndarray, foreground_mask: np.ndarray, text_mask: np.ndarray) -> bool:
    if not np.any(text_mask):
        return False
    component_area = int(np.count_nonzero(component_mask))
    if component_area == 0:
        return True
    text_area = int(np.count_nonzero((component_mask > 0) & (text_mask > 0)))
    if text_area == 0:
        return False
    non_text_foreground = int(np.count_nonzero((component_mask > 0) & (foreground_mask > 0) & (text_mask == 0)))
    return non_text_foreground < max(20, int(component_area * 0.08))


def _claim_padding(component: Component) -> int:
    shorter_side = max(1.0, min(component.bbox.width, component.bbox.height))
    if component.type == "text":
        return max(2, int(round(shorter_side * 0.12)))
    return max(1, int(round(shorter_side * 0.04)))


def _save_masked_crop_asset(
    store: ProjectStore,
    project_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    bbox: BBox,
    name: str,
) -> str | None:
    try:
        x1 = max(0, round(bbox.x))
        y1 = max(0, round(bbox.y))
        x2 = min(image.shape[1], round(bbox.x + bbox.width))
        y2 = min(image.shape[0], round(bbox.y + bbox.height))
        crop_bgr = image[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
        alpha = mask[y1:y2, x1:x2]
        crop[:, :, 3] = alpha
        path = store.asset_dir(project_id) / f"{name}.png"
        Image.fromarray(crop).save(path)
        return str(path)
    except Exception:
        return None


def _save_mask_asset(store: ProjectStore, project_id: str, mask: Any, index: int, prompt: str) -> str | None:
    try:
        array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
        if array.ndim > 2:
            array = array.squeeze()
        image = Image.fromarray((array > 0).astype("uint8") * 255, mode="L")
        path = store.asset_dir(project_id) / f"mask-{prompt}-{index}.png"
        image.save(path)
        return str(path)
    except Exception:
        return None
