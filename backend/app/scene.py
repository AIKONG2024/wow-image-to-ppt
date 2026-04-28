from __future__ import annotations

import base64
import html
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .image_editing import erase_regions
from .models import BBox, Project, PptPrimitive, SceneGraph, SceneNode
from .reconstruction import reconstruct_project


OCR_TEXT_REPLACEMENTS = (
    ("핵심 매시지", "핵심 메시지"),
    ("급감 시정", "급감 시점"),
    ("발생 시정", "발생 시점"),
    ("나빳", "나빴"),
    ("S0 S", "So"),
    ("A)(", "A)"),
    ("B HOW", "B) HOW"),
    ("D] ", "D) "),
)
SOURCE_CROP_VISUAL_TYPES = {"chart", "table", "diagram"}


def build_scene_graph(project: Project, source: Image.Image) -> SceneGraph:
    primitives = reconstruct_project(project, source)
    nodes = [_primitive_to_scene_node(primitive) for primitive in primitives]
    nodes = _remove_redundant_header_images(nodes, project.width, project.height)
    _mark_text_colors(nodes, source)
    nodes = _merge_adjacent_text_line_nodes(nodes)
    nodes = _trim_text_nodes_at_large_side_images(nodes, project.width, project.height)
    _normalize_dark_label_rects(nodes)
    _trim_chart_images_below_overlapping_text_labels(nodes)
    nodes = _preserve_complex_shape_regions(nodes, source)
    nodes.extend(_synthesized_info_card_rects(nodes, project.width, project.height))
    nodes.extend(_synthesized_text_background_rects(nodes, source))
    nodes.extend(_synthesized_bullet_texts(nodes, source))
    _mark_text_regions_for_picture_erasing(nodes)
    return SceneGraph(width=project.width, height=project.height, nodes=nodes)


def _trim_text_nodes_at_large_side_images(nodes: list[SceneNode], width: int, height: int) -> list[SceneNode]:
    side_images = [
        node
        for node in nodes
        if node.kind == "image"
        and node.source_component_type == "image"
        and _is_large_side_image(node.bbox, width, height)
    ]
    if not side_images:
        return nodes
    trimmed: list[SceneNode] = []
    for node in nodes:
        if node.kind != "text" or not node.text:
            trimmed.append(node)
            continue
        bbox = node.bbox
        for image_node in side_images:
            if image_node.bbox.x <= bbox.x:
                continue
            if _vertical_overlap_ratio(bbox, image_node.bbox) < 0.12:
                continue
            overlap = _horizontal_intersection(bbox, image_node.bbox)
            if overlap <= 0 or overlap / max(1.0, bbox.width) > 0.25:
                continue
            new_width = image_node.bbox.x - bbox.x - 6.0
            if new_width >= bbox.width * 0.55:
                bbox = BBox(x=bbox.x, y=bbox.y, width=max(1.0, new_width), height=bbox.height)
        if bbox == node.bbox:
            trimmed.append(node)
        else:
            trimmed.append(node.model_copy(update={"bbox": bbox}))
    return trimmed


def _merge_adjacent_text_line_nodes(nodes: list[SceneNode]) -> list[SceneNode]:
    text_nodes = [node for node in nodes if node.kind == "text" and node.text]
    if len(text_nodes) < 2:
        return nodes
    other_nodes = [node for node in nodes if not (node.kind == "text" and node.text)]
    rows: list[list[SceneNode]] = []
    for node in sorted(text_nodes, key=lambda item: (item.bbox.y + item.bbox.height / 2, item.bbox.x)):
        for row in rows:
            if _same_text_row(row[0].bbox, node.bbox):
                row.append(node)
                break
        else:
            rows.append([node])

    merged: list[SceneNode] = []
    for row in rows:
        row.sort(key=lambda item: item.bbox.x)
        current: list[SceneNode] = []
        for node in row:
            if not current:
                current = [node]
                continue
            previous = current[-1]
            gap = node.bbox.x - (previous.bbox.x + previous.bbox.width)
            max_gap = max(28.0, min(previous.bbox.height, node.bbox.height) * 1.15)
            if gap <= max_gap and _similar_text_color(previous.text_color, node.text_color) and _should_merge_text_neighbors(previous, node):
                current.append(node)
            else:
                merged.append(_merge_text_run(current))
                current = [node]
        if current:
            merged.append(_merge_text_run(current))
    return [*other_nodes, *merged]


def _same_text_row(left: BBox, right: BBox) -> bool:
    center_delta = abs((left.y + left.height / 2) - (right.y + right.height / 2))
    height = max(1.0, min(left.height, right.height))
    return center_delta <= max(5.0, height * 0.45) and _vertical_overlap_ratio(left, right) >= 0.45


def _similar_text_color(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return True
    return abs(_hex_luma(left) - _hex_luma(right)) <= 80


def _should_merge_text_neighbors(left: SceneNode, right: SceneNode) -> bool:
    if _is_large_number_label(left, right) or _is_large_number_label(right, left):
        return False
    return True


def _is_large_number_label(candidate: SceneNode, neighbor: SceneNode) -> bool:
    text = (candidate.text or "").strip()
    return (
        text.isdigit()
        and len(text) <= 2
        and candidate.bbox.height >= neighbor.bbox.height * 1.6
        and candidate.bbox.width <= max(48.0, neighbor.bbox.height * 2.4)
    )


def _merge_text_run(nodes: list[SceneNode]) -> SceneNode:
    if len(nodes) == 1:
        return nodes[0]
    bbox = _bbox_union([node.bbox for node in nodes])
    text = " ".join((node.text or "").strip() for node in nodes if (node.text or "").strip())
    return SceneNode(
        id=f"text-line-{uuid.uuid4().hex[:10]}",
        kind="text",
        bbox=bbox,
        source_component_id="+".join(node.source_component_id or node.id for node in nodes),
        source_component_type="text",
        text=_normalize_ocr_text(text),
        text_color=nodes[0].text_color,
        z_index=max(node.z_index for node in nodes),
    )


def render_scene_svg(scene: SceneGraph, source: Image.Image) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{scene.width}" height="{scene.height}" viewBox="0 0 {scene.width} {scene.height}">',
        "<defs>",
        '<marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">',
        '<path d="M0,0 L8,3 L0,6 Z" fill="#111827"/>',
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="{scene.width}" height="{scene.height}" fill="#ffffff"/>',
    ]
    for node in scene_paint_order(scene.nodes):
        if node.kind == "rect":
            parts.append(_rect_svg(node))
        elif node.kind == "image":
            image_data = _image_data_uri(source, node)
            if image_data:
                parts.append(_image_svg(node, image_data))
        elif node.kind == "text" and node.text:
            parts.append(_text_svg(node))
        elif node.kind in {"line", "arrow"}:
            parts.append(_line_svg(node))
    parts.append("</svg>")
    return "\n".join(parts)


def scene_paint_order(nodes: list[SceneNode]) -> list[SceneNode]:
    return sorted(nodes, key=lambda item: (item.kind == "text", item.z_index, item.bbox.y, item.bbox.x))


def _primitive_to_scene_node(primitive: PptPrimitive) -> SceneNode:
    kind_map = {
        "textbox": "text",
        "shape": "rect",
        "picture": "image",
        "line": "line",
        "arrow": "arrow",
    }
    return SceneNode(
        id=primitive.id,
        kind=kind_map[primitive.kind],
        bbox=primitive.bbox,
        source_component_id=primitive.source_component_id,
        source_component_type=primitive.source_component_type,
        text=_normalize_ocr_text(primitive.text),
        text_color=primitive.text_color,
        fill_color=primitive.fill_color,
        line_color=primitive.line_color,
        line_width=primitive.line_width,
        asset_path=primitive.asset_path,
        mask_path=primitive.mask_path,
        x1=primitive.x1,
        y1=primitive.y1,
        x2=primitive.x2,
        y2=primitive.y2,
        z_index=primitive.z_index,
    )


def _normalize_ocr_text(text: str | None) -> str | None:
    if not text:
        return text
    normalized = text
    for source, replacement in OCR_TEXT_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    return normalized


def _remove_redundant_header_images(nodes: list[SceneNode], width: int, height: int) -> list[SceneNode]:
    images = [node for node in nodes if node.kind == "image"]
    redundant: set[str] = set()
    for parent in images:
        if not _is_header_picture(parent.bbox, width, height):
            continue
        for child in images:
            if child.id == parent.id:
                continue
            child_area = _bbox_area(child.bbox)
            parent_area = _bbox_area(parent.bbox)
            if child_area < parent_area * 0.2:
                continue
            if _containment_ratio(child.bbox, parent.bbox) >= 0.9:
                redundant.add(child.id)
    return [node for node in nodes if node.id not in redundant]


def _mark_text_regions_for_picture_erasing(nodes: list[SceneNode]) -> None:
    text_nodes = [node for node in nodes if node.kind == "text" and node.text]
    scene_width = _scene_width(nodes)
    scene_height = _scene_height(nodes)
    for image_node in nodes:
        if image_node.kind != "image":
            continue
        if image_node.source_component_type == "icon":
            continue
        protect_large_art_text = image_node.source_component_type == "image" and _is_large_side_image(
            image_node.bbox, scene_width, scene_height
        )
        image_node.erase_boxes = [
            text_node.bbox
            for text_node in text_nodes
            if _should_erase_text_from_image(
                image_node.bbox, text_node.bbox, protect_large_art_text=protect_large_art_text
            )
        ]


def _should_erase_text_from_image(
    image_bbox: BBox, text_bbox: BBox, *, protect_large_art_text: bool = False
) -> bool:
    intersection = _intersection_area(text_bbox, image_bbox)
    if intersection <= 0:
        return False
    text_area = _bbox_area(text_bbox)
    image_area = _bbox_area(image_bbox)
    if text_area <= 0 or image_area <= 0:
        return False
    text_covered = intersection / text_area
    image_covered = intersection / image_area
    if protect_large_art_text and text_area / image_area > 0.14:
        return False
    return text_covered >= 0.55 and image_covered <= 0.55


def _scene_width(nodes: list[SceneNode]) -> int:
    if not nodes:
        return 1
    return max(1, int(round(max(node.bbox.x + node.bbox.width for node in nodes))))


def _scene_height(nodes: list[SceneNode]) -> int:
    if not nodes:
        return 1
    return max(1, int(round(max(node.bbox.y + node.bbox.height for node in nodes))))


def _preserve_complex_shape_regions(nodes: list[SceneNode], source: Image.Image) -> list[SceneNode]:
    candidates: list[SceneNode] = []
    for node in nodes:
        if not (
            node.kind == "rect"
            and node.source_component_type == "shape"
            and node.z_index <= 1200
            and not _is_top_header_region(node.bbox, source.width, source.height)
            and _is_complex_source_region(source, node.bbox)
            and not _contains_large_chart_image(node.bbox, nodes)
        ):
            continue
        candidates.append(
            SceneNode(
                id=f"complex-region-{uuid.uuid4().hex[:10]}",
                kind="image",
                bbox=node.bbox,
                source_component_id=node.source_component_id,
                source_component_type="image",
                z_index=1750,
            )
        )
    return _replace_regions_with_source_images(nodes, candidates)


def _replace_regions_with_source_images(nodes: list[SceneNode], regions: list[SceneNode]) -> list[SceneNode]:
    if not regions:
        return nodes
    regions = _dedupe_preserved_regions(regions)
    removed: set[str] = set()
    for region in regions:
        for node in nodes:
            if node.id == region.id:
                continue
            if node.kind in {"line", "arrow"}:
                continue
            if node.kind != "text" and _containment_ratio(node.bbox, region.bbox) >= 0.68:
                removed.add(node.id)
    kept = [node for node in nodes if node.id not in removed]
    return [*kept, *regions]


def _dedupe_preserved_regions(regions: list[SceneNode]) -> list[SceneNode]:
    kept: list[SceneNode] = []
    for candidate in sorted(regions, key=lambda item: _bbox_area(item.bbox), reverse=True):
        if any(_containment_ratio(candidate.bbox, existing.bbox) >= 0.82 for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _contains_large_chart_image(bbox: BBox, nodes: list[SceneNode]) -> bool:
    for node in nodes:
        if node.kind != "image" or node.source_component_type not in {"chart", "table"}:
            continue
        if _bbox_area(node.bbox) < _bbox_area(bbox) * 0.35:
            continue
        if _containment_ratio(node.bbox, bbox) >= 0.72:
            return True
    return False


def _is_large_side_image(bbox: BBox, width: int, height: int) -> bool:
    return bbox.height >= height * 0.55 and bbox.width >= width * 0.18


def _is_top_header_region(bbox: BBox, width: int, height: int) -> bool:
    return bbox.y <= height * 0.04 and bbox.width >= width * 0.55 and bbox.height <= height * 0.28


def _is_complex_source_region(source: Image.Image, bbox: BBox) -> bool:
    if _bbox_area(bbox) < 1800:
        return False
    crop = _crop(source, bbox).convert("RGB")
    if crop.width < 18 or crop.height < 18:
        return False
    rgb = np.asarray(crop, dtype=np.int16)
    pixels = rgb.reshape(-1, 3)
    luma = _pixel_luma(pixels)
    channel_spread = pixels.max(axis=1) - pixels.min(axis=1)
    nonwhite = (np.max(np.abs(pixels - 255), axis=1) > 18) | (channel_spread > 14)
    nonwhite_ratio = float(np.count_nonzero(nonwhite)) / max(1, len(pixels))
    if nonwhite_ratio < 0.16:
        return False
    quantized = (pixels // 32) * 32
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant_ratio = float(counts.max()) / max(1, len(pixels))
    prominent_colors = int(np.count_nonzero(counts >= max(12, int(len(pixels) * 0.012))))
    color_span = int(max(pixels.max(axis=0) - pixels.min(axis=0)))
    bright_ratio = float(np.count_nonzero(luma > 235)) / max(1, len(luma))
    dark_ratio = float(np.count_nonzero(luma < 80)) / max(1, len(luma))
    return (
        prominent_colors >= 3
        and color_span >= 70
        and dominant_ratio <= 0.78
        and (bright_ratio >= 0.08 or dark_ratio >= 0.08)
    )


def _mark_text_colors(nodes: list[SceneNode], source: Image.Image) -> None:
    dark_rects = [
        node
        for node in nodes
        if node.kind == "rect" and node.fill_color is not None and _hex_luma(node.fill_color) <= 120
    ]
    for node in nodes:
        if node.kind == "text" and node.text:
            node.text_color = _estimate_text_color(source, node.bbox)
            if _hex_luma(node.text_color) < 140 and _has_dark_containing_rect(node.bbox, dark_rects):
                node.text_color = "FFFFFF"


def _normalize_dark_label_rects(nodes: list[SceneNode]) -> None:
    text_nodes = [node for node in nodes if node.kind == "text" and node.text]
    for rect in nodes:
        if rect.kind != "rect" or not rect.fill_color:
            continue
        if _hex_luma(rect.fill_color) > 95 or rect.bbox.height > 58:
            continue
        overlapping_texts = [
            text
            for text in text_nodes
            if _intersection_area(text.bbox, rect.bbox) >= _bbox_area(text.bbox) * 0.45
            and _hex_luma(text.text_color) > 140
        ]
        if not overlapping_texts:
            continue
        text_box = _bbox_union([text.bbox for text in overlapping_texts])
        if rect.bbox.width <= text_box.width * 1.4:
            continue
        if rect.bbox.height > max(58.0, text_box.height * 2.2):
            continue
        pad_x = max(18.0, text_box.height * 0.75)
        x1 = max(rect.bbox.x, text_box.x - pad_x)
        x2 = min(rect.bbox.x + rect.bbox.width, text_box.x + text_box.width + pad_x)
        if x2 - x1 < text_box.width:
            continue
        rect.bbox = BBox(x=x1, y=rect.bbox.y, width=x2 - x1, height=rect.bbox.height)


def _trim_chart_images_below_overlapping_text_labels(nodes: list[SceneNode]) -> None:
    text_nodes = [node for node in nodes if node.kind == "text" and node.text]
    for image_node in nodes:
        if image_node.kind != "image" or image_node.source_component_type not in SOURCE_CROP_VISUAL_TYPES:
            continue
        image_top = image_node.bbox.y
        image_bottom = image_node.bbox.y + image_node.bbox.height
        trim_to = image_top
        for text_node in text_nodes:
            text_bottom = text_node.bbox.y + text_node.bbox.height
            if not (text_node.bbox.y < image_top < text_bottom):
                continue
            horizontal_overlap = _horizontal_intersection(text_node.bbox, image_node.bbox)
            if horizontal_overlap < min(text_node.bbox.width * 0.45, image_node.bbox.width * 0.3):
                continue
            trim_to = max(trim_to, text_bottom)
        if trim_to > image_top and trim_to < image_bottom - 8:
            image_node.bbox = BBox(
                x=image_node.bbox.x,
                y=trim_to,
                width=image_node.bbox.width,
                height=image_bottom - trim_to,
            )


def _synthesized_text_background_rects(nodes: list[SceneNode], source: Image.Image) -> list[SceneNode]:
    rects: list[SceneNode] = []
    existing_rects = [node for node in nodes if node.kind == "rect"]
    for text_node in nodes:
        if text_node.kind != "text" or _hex_luma(text_node.text_color) < 180:
            continue
        if text_node.bbox.y < source.height * 0.14:
            continue
        if _has_dark_containing_rect(text_node.bbox, existing_rects):
            continue
        background = _detect_text_background_rect(text_node, source)
        if background is not None:
            rects.append(background)
            existing_rects.append(background)
    return rects


def _synthesized_bullet_texts(nodes: list[SceneNode], source: Image.Image) -> list[SceneNode]:
    bullets: list[SceneNode] = []
    text_nodes = [node for node in nodes if node.kind == "text" and node.text]
    for text_node in text_nodes:
        stripped = text_node.text.strip()
        if not stripped or stripped.startswith(("•", "-", "·")):
            continue
        if text_node.bbox.x < 24 or text_node.bbox.height < 12:
            continue
        search_left = max(0.0, text_node.bbox.x - max(46.0, text_node.bbox.height * 3.2))
        search_right = max(search_left, text_node.bbox.x - max(8.0, text_node.bbox.height * 0.45))
        search = BBox(
            x=search_left,
            y=max(0.0, text_node.bbox.y + text_node.bbox.height * 0.15),
            width=search_right - search_left,
            height=text_node.bbox.height * 0.7,
        )
        bullet_box = _detect_bullet_dot(source, search, text_node.bbox)
        if bullet_box is None:
            continue
        bullets.append(
            SceneNode(
                id=f"bullet-{uuid.uuid4().hex[:10]}",
                kind="text",
                bbox=bullet_box,
                source_component_id=text_node.source_component_id,
                source_component_type="text",
                text="•",
                text_color="111111",
                z_index=3000,
            )
        )
    return bullets


def _detect_bullet_dot(source: Image.Image, search: BBox, text_bbox: BBox) -> BBox | None:
    crop = _crop(source, search).convert("RGB")
    if crop.width == 0 or crop.height == 0:
        return None
    rgb = np.asarray(crop, dtype=np.int16)
    luma = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    dark_mask = luma < 70
    if int(np.count_nonzero(dark_mask)) < 8:
        return None
    rows, columns = np.where(dark_mask)
    local_x1 = int(columns.min())
    local_x2 = int(columns.max()) + 1
    local_y1 = int(rows.min())
    local_y2 = int(rows.max()) + 1
    width = float(local_x2 - local_x1)
    height = float(local_y2 - local_y1)
    if width < 3 or height < 3 or width > 18 or height > 18:
        return None
    aspect = width / max(height, 1.0)
    if aspect < 0.55 or aspect > 1.8:
        return None
    center_y = search.y + local_y1 + height / 2
    if abs(center_y - (text_bbox.y + text_bbox.height / 2)) > text_bbox.height * 0.45:
        return None
    return BBox(
        x=search.x + local_x1 - max(1.0, text_bbox.height * 0.08),
        y=text_bbox.y,
        width=max(10.0, text_bbox.height * 0.7),
        height=text_bbox.height,
    )


def _synthesized_info_card_rects(nodes: list[SceneNode], width: int, height: int) -> list[SceneNode]:
    icon_rects = [
        node
        for node in nodes
        if node.kind == "rect"
        and node.fill_color
        and node.line_color
        and node.bbox.width <= width * 0.16
        and node.bbox.height >= max(24, height * 0.05)
        and _bbox_area(node.bbox) <= width * height * 0.04
    ]
    icon_rects.sort(key=lambda node: node.bbox.x)
    text_nodes = [node for node in nodes if node.kind == "text" and node.text]
    cards: list[SceneNode] = []
    for index, icon_rect in enumerate(icon_rects):
        if _has_larger_card_rect(icon_rect, nodes):
            continue
        next_icon = icon_rects[index + 1] if index + 1 < len(icon_rects) else None
        group_texts = _nearby_card_texts(icon_rect, next_icon, text_nodes)
        if not group_texts:
            continue
        boxes = [icon_rect.bbox, *[node.bbox for node in group_texts]]
        union = _bbox_union(boxes)
        left_pad = _info_card_left_padding(icon_rect.bbox)
        right_pad = max(8.0, icon_rect.bbox.width * 0.18)
        pad_y = max(8.0, icon_rect.bbox.height * 0.18)
        bbox = _clip_bbox(_expanded_bbox_asymmetric(union, left_pad, right_pad, pad_y, pad_y), width, height)
        cards.append(
            SceneNode(
                id=f"info-card-{uuid.uuid4().hex[:10]}",
                kind="rect",
                bbox=bbox,
                source_component_id=icon_rect.source_component_id,
                source_component_type="shape",
                fill_color=icon_rect.fill_color,
                line_color=icon_rect.line_color,
                line_width=1.25,
                z_index=950,
            )
        )
    return cards


def _has_larger_card_rect(icon_rect: SceneNode, nodes: list[SceneNode]) -> bool:
    for node in nodes:
        if node.id == icon_rect.id or node.kind != "rect":
            continue
        if not node.fill_color or not node.line_color:
            continue
        if _bbox_area(node.bbox) <= _bbox_area(icon_rect.bbox) * 1.8:
            continue
        if _containment_ratio(icon_rect.bbox, node.bbox) >= 0.75:
            return True
    return False


def _nearby_card_texts(icon_rect: SceneNode, next_icon: SceneNode | None, text_nodes: list[SceneNode]) -> list[SceneNode]:
    bbox = icon_rect.bbox
    lower = bbox.y - max(10.0, bbox.height * 0.2)
    upper = bbox.y + bbox.height + max(34.0, bbox.height * 0.6)
    left = bbox.x + bbox.width * 0.55
    right = bbox.x + 260.0
    if next_icon is not None:
        right = min(right, next_icon.bbox.x + next_icon.bbox.width + 12)
    result = []
    for text in text_nodes:
        if text.bbox.y + text.bbox.height < lower or text.bbox.y > upper:
            continue
        if text.bbox.x < left or text.bbox.x > right:
            continue
        result.append(text)
    return result


def _info_card_left_padding(icon_bbox: BBox) -> float:
    narrow_icon_extra = max(0.0, icon_bbox.height - icon_bbox.width) + 2.0
    return max(8.0, icon_bbox.width * 0.28, narrow_icon_extra)


def _has_dark_containing_rect(bbox: BBox, rects: list[SceneNode]) -> bool:
    for rect in rects:
        if rect.fill_color is None or _hex_luma(rect.fill_color) > 120:
            continue
        if _containment_ratio(bbox, rect.bbox) >= 0.72:
            return True
    return False


def _detect_text_background_rect(text_node: SceneNode, source: Image.Image) -> SceneNode | None:
    search = _clip_bbox(_expanded_bbox(text_node.bbox, 24, 10), source.width, source.height)
    crop = _crop(source, search).convert("RGB")
    rgb = np.asarray(crop, dtype=np.int16)
    luma = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    dark_mask = luma < 95
    if int(np.count_nonzero(dark_mask)) < max(20, int(rgb.shape[0] * rgb.shape[1] * 0.05)):
        return None
    rows, columns = np.where(dark_mask)
    local_x1 = int(columns.min())
    local_x2 = int(columns.max()) + 1
    local_y1 = int(rows.min())
    local_y2 = int(rows.max()) + 1
    bbox = BBox(
        x=search.x + local_x1,
        y=search.y + local_y1,
        width=float(local_x2 - local_x1),
        height=float(local_y2 - local_y1),
    )
    if bbox.width < text_node.bbox.width * 0.85 or bbox.height < text_node.bbox.height * 0.75:
        return None
    if bbox.width < text_node.bbox.width * 1.08 or bbox.height < text_node.bbox.height * 1.18:
        return None
    if _intersection_area(text_node.bbox, bbox) < _bbox_area(text_node.bbox) * 0.45:
        return None
    fill_color = _rgb_to_hex(_dominant_color(rgb.reshape(-1, 3)[dark_mask.reshape(-1)]))
    return SceneNode(
        id=f"text-bg-{uuid.uuid4().hex[:10]}",
        kind="rect",
        bbox=bbox,
        source_component_id=text_node.source_component_id,
        source_component_type="shape",
        fill_color=fill_color,
        line_color=None,
        line_width=None,
        z_index=2900,
    )


def _estimate_text_color(source: Image.Image, bbox: BBox) -> str:
    crop = _crop(source, bbox).convert("RGB")
    if crop.width == 0 or crop.height == 0:
        return "111827"
    rgb = np.asarray(crop, dtype=np.int16)
    local_dark_text = _light_foreground_on_dark_local_background(rgb)
    if local_dark_text is not None:
        return _rgb_to_hex(local_dark_text)
    background = np.array(_estimate_background_color(source, bbox), dtype=np.int16)
    distance = np.max(np.abs(rgb - background), axis=2)
    foreground = rgb[distance > 34]
    if len(foreground) < max(6, int(rgb.shape[0] * rgb.shape[1] * 0.015)):
        background_luma = _luma(background)
        return "111827" if background_luma > 150 else "FFFFFF"
    background_luma = _luma(background)
    if background_luma < 150:
        bright = foreground[_pixel_luma(foreground) > 185]
        if len(bright) >= max(4, int(rgb.shape[0] * rgb.shape[1] * 0.004)):
            color = _dominant_color(bright)
            return _rgb_to_hex(color)
    color = _contrasting_foreground_color(foreground, background)
    return _rgb_to_hex(tuple(int(round(channel)) for channel in color))


def _light_foreground_on_dark_local_background(rgb: np.ndarray) -> tuple[int, int, int] | None:
    pixels = rgb.reshape(-1, 3)
    luma = _pixel_luma(pixels)
    dark_ratio = float(np.count_nonzero(luma < 95)) / max(1, len(luma))
    border_luma = _pixel_luma(_border_pixels(rgb))
    border_dark_ratio = float(np.count_nonzero(border_luma < 95)) / max(1, len(border_luma))
    bright = pixels[luma > 175]
    bright_ratio = float(len(bright)) / max(1, len(luma))
    if dark_ratio < 0.48 or border_dark_ratio < 0.45 or bright_ratio < 0.025:
        return None
    color = _dominant_color(bright)
    if _luma(np.array(color, dtype=np.int16)) < 150:
        return None
    return color


def _estimate_background_color(source: Image.Image, bbox: BBox) -> tuple[int, int, int]:
    sample_bbox = _clip_bbox(
        _expanded_bbox(
            bbox,
            max(4.0, bbox.width * 0.12),
            max(3.0, bbox.height * 0.35),
        ),
        source.width,
        source.height,
    )
    sample = _crop(source, sample_bbox).convert("RGB")
    pixels = np.asarray(sample, dtype=np.int16)
    if pixels.size == 0:
        return (255, 255, 255)
    border = _border_pixels(pixels)
    if len(border) >= 12:
        return _dominant_color(border)
    return _dominant_color(pixels.reshape(-1, 3))


def _border_pixels(pixels: np.ndarray) -> np.ndarray:
    height, width = pixels.shape[:2]
    if height == 0 or width == 0:
        return pixels.reshape(-1, 3)
    thickness = max(1, min(height, width) // 8)
    parts = [
        pixels[:thickness, :, :],
        pixels[-thickness:, :, :],
        pixels[:, :thickness, :],
        pixels[:, -thickness:, :],
    ]
    return np.concatenate([part.reshape(-1, 3) for part in parts], axis=0)


def _dominant_color(pixels: np.ndarray) -> tuple[int, int, int]:
    if len(pixels) > 30000:
        stride = max(1, len(pixels) // 30000)
        pixels = pixels[::stride]
    quantized = (pixels // 12) * 12
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    winner = colors[int(np.argmax(counts))]
    bucket = np.all(quantized == winner, axis=1)
    representative = np.median(pixels[bucket] if np.any(bucket) else pixels, axis=0)
    return tuple(int(round(channel)) for channel in representative)


def _contrasting_foreground_color(pixels: np.ndarray, background: np.ndarray) -> tuple[int, int, int]:
    quantized = (pixels // 12) * 12
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    min_count = max(3, int(len(pixels) * 0.015))
    best_index = 0
    best_score = -1.0
    for index, color in enumerate(colors):
        if counts[index] < min_count:
            continue
        contrast = float(np.max(np.abs(color.astype(np.int16) - background)))
        score = contrast * (1.0 + min(float(counts[index]), 80.0) / 80.0)
        if score > best_score:
            best_score = score
            best_index = index
    winner = colors[best_index]
    bucket = np.all(quantized == winner, axis=1)
    representative = np.median(pixels[bucket] if np.any(bucket) else pixels, axis=0)
    return tuple(int(round(channel)) for channel in representative)


def _is_header_picture(bbox: BBox, width: int, height: int) -> bool:
    return bbox.x <= 3 and bbox.y <= 3 and bbox.width >= width * 0.85 and bbox.height <= height * 0.25


def _rect_svg(node: SceneNode) -> str:
    bbox = node.bbox
    fill = _svg_color(node.fill_color, "none")
    stroke = _svg_color(node.line_color, "none")
    width = node.line_width if node.line_width is not None else 0
    return (
        f'<rect id="{html.escape(node.id)}" {_source_attr(node)} '
        f'x="{_fmt(bbox.x)}" y="{_fmt(bbox.y)}" width="{_fmt(bbox.width)}" height="{_fmt(bbox.height)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{_fmt(width)}"/>'
    )


def _image_svg(node: SceneNode, data_uri: str) -> str:
    bbox = node.bbox
    return (
        f'<image id="{html.escape(node.id)}" {_source_attr(node)} '
        f'x="{_fmt(bbox.x)}" y="{_fmt(bbox.y)}" width="{_fmt(bbox.width)}" height="{_fmt(bbox.height)}" '
        f'href="{data_uri}" preserveAspectRatio="none"/>'
    )


def _text_svg(node: SceneNode) -> str:
    bbox = node.bbox
    font_size = _svg_text_font_size(bbox, node.text or "")
    font_family = _svg_text_font_family(bbox, node.text or "")
    font_weight = _svg_text_font_weight(bbox, node.text or "")
    font_style = _svg_text_font_style(bbox, node.text or "")
    y = bbox.y + bbox.height * 0.72
    escaped_text = html.escape(node.text or "")
    text_fit = f' textLength="{_fmt(bbox.width)}" lengthAdjust="spacingAndGlyphs"' if _is_latin_display_text(bbox, node.text or "") else ""
    return (
        f'<text id="{html.escape(node.id)}" {_source_attr(node)} '
        f'x="{_fmt(bbox.x)}" y="{_fmt(y)}"{text_fit} '
        f'font-family="{font_family}" font-size="{_fmt(font_size)}" '
        f'font-weight="{font_weight}" font-style="{font_style}" '
        f'fill="{_svg_color(node.text_color, "#111827")}">{escaped_text}</text>'
    )


def _line_svg(node: SceneNode) -> str:
    bbox = node.bbox
    x1 = node.x1 if node.x1 is not None else bbox.x
    y1 = node.y1 if node.y1 is not None else bbox.y
    x2 = node.x2 if node.x2 is not None else bbox.x + bbox.width
    y2 = node.y2 if node.y2 is not None else bbox.y + bbox.height
    color = _svg_color(node.line_color, "#111827")
    width = node.line_width if node.line_width is not None else 1.25
    marker = ' marker-end="url(#arrowhead)"' if node.kind == "arrow" else ""
    return (
        f'<line id="{html.escape(node.id)}" {_source_attr(node)} '
        f'x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
        f'stroke="{color}" stroke-width="{_fmt(width)}"{marker}/>'
    )


def _image_data_uri(source: Image.Image, node: SceneNode) -> str | None:
    asset = Path(node.asset_path) if node.asset_path else None
    if asset and asset.exists() and not _should_use_source_crop(node):
        with Image.open(asset) as opened:
            image = opened.convert("RGBA")
    else:
        image = _crop(source, node.bbox).convert("RGBA")
        if image.width == 0 or image.height == 0:
            return None
        if node.mask_path and Path(node.mask_path).exists() and not _should_use_source_crop(node):
            image = _apply_mask(image, Path(node.mask_path), node.bbox)
    image = erase_regions(image, node.bbox, node.erase_boxes)
    stream = BytesIO()
    image.save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _should_use_source_crop(node: SceneNode) -> bool:
    return node.source_component_type in SOURCE_CROP_VISUAL_TYPES


def _svg_text_font_size(bbox: BBox, text: str) -> float:
    longest = max(text.splitlines() or [text], key=len)
    if _is_latin_display_text(bbox, text):
        by_width = bbox.width / max(_text_units(longest, latin_width=0.36), 1.0) * 0.94
        by_height = bbox.height * 0.86
        return max(12.0, min(110.0, by_width, by_height))
    by_width = bbox.width / max(_text_units(longest), 1.0) * 0.95
    by_height = bbox.height * 0.72
    return max(8.0, min(96.0, by_width, by_height))


def _svg_text_font_family(bbox: BBox, text: str) -> str:
    if _is_latin_display_text(bbox, text):
        return "Impact, Haettenschweiler, Arial Black, Arial Narrow, sans-serif"
    if _is_latin_text(text):
        return "Arial, Calibri, sans-serif"
    return "Malgun Gothic, Arial, sans-serif"


def _svg_text_font_weight(bbox: BBox, text: str) -> str:
    if _is_latin_display_text(bbox, text):
        return "900"
    if _is_short_uppercase_label(text):
        return "700"
    return "400"


def _svg_text_font_style(bbox: BBox, text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(("Narrative,", "A narrative", "The narrative")):
        return "italic"
    return "normal"


def _text_units(text: str, latin_width: float = 0.6) -> float:
    total = 0.0
    for char in text:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3 or 0x4E00 <= code <= 0x9FFF:
            total += 1.0
        elif char.isspace():
            total += 0.35
        else:
            total += latin_width
    return total


def _is_latin_display_text(bbox: BBox, text: str) -> bool:
    stripped = text.strip()
    return (
        bbox.height >= 48
        and len(stripped) >= 12
        and _latin_character_ratio(stripped) >= 0.72
        and any(char.isupper() for char in stripped)
    )


def _is_latin_text(text: str) -> bool:
    return _latin_character_ratio(text.strip()) >= 0.65


def _is_short_uppercase_label(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    uppercase = sum(1 for char in letters if char.upper() == char)
    return len(text.strip()) <= 56 and uppercase / max(1, len(letters)) >= 0.75


def _latin_character_ratio(text: str) -> float:
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return 0.0
    latin = sum(1 for char in non_space if ord(char) < 128)
    return latin / len(non_space)


def _apply_mask(crop: Image.Image, mask_path: Path, bbox: BBox) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    mask_crop = mask.crop(
        (
            max(0, round(bbox.x)),
            max(0, round(bbox.y)),
            min(mask.width, round(bbox.x + bbox.width)),
            min(mask.height, round(bbox.y + bbox.height)),
        )
    ).resize(crop.size)
    crop.putalpha(mask_crop)
    return crop


def _crop(source: Image.Image, bbox: BBox) -> Image.Image:
    return source.crop(
        (
            max(0, round(bbox.x)),
            max(0, round(bbox.y)),
            min(source.width, round(bbox.x + bbox.width)),
            min(source.height, round(bbox.y + bbox.height)),
        )
    )


def _source_attr(node: SceneNode) -> str:
    if not node.source_component_id:
        return ""
    return f'data-source="{html.escape(node.source_component_id)}"'


def _svg_color(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    return f"#{value.strip().lstrip('#')}"


def _containment_ratio(inner: BBox, outer: BBox) -> float:
    return _intersection_area(inner, outer) / max(1.0, inner.width * inner.height)


def _intersection_area(left: BBox, right: BBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _horizontal_intersection(left: BBox, right: BBox) -> float:
    x1 = max(left.x, right.x)
    x2 = min(left.x + left.width, right.x + right.width)
    return max(0.0, x2 - x1)


def _vertical_overlap_ratio(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    return overlap / max(1.0, min(left.height, right.height))


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox.width * bbox.height)


def _bbox_union(boxes: list[BBox]) -> BBox:
    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.x + box.width for box in boxes)
    y2 = max(box.y + box.height for box in boxes)
    return BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def _expanded_bbox(bbox: BBox, horizontal: float, vertical: float) -> BBox:
    return BBox(
        x=bbox.x - horizontal,
        y=bbox.y - vertical,
        width=bbox.width + horizontal * 2,
        height=bbox.height + vertical * 2,
    )


def _expanded_bbox_asymmetric(bbox: BBox, left: float, right: float, top: float, bottom: float) -> BBox:
    return BBox(
        x=bbox.x - left,
        y=bbox.y - top,
        width=bbox.width + left + right,
        height=bbox.height + top + bottom,
    )


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1 = max(0.0, bbox.x)
    y1 = max(0.0, bbox.y)
    x2 = min(float(width), bbox.x + bbox.width)
    y2 = min(float(height), bbox.y + bbox.height)
    return BBox(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1))


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    red, green, blue = color
    return f"{red:02X}{green:02X}{blue:02X}"


def _luma(color: np.ndarray) -> float:
    return float(color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722)


def _pixel_luma(pixels: np.ndarray) -> np.ndarray:
    return pixels[:, 0] * 0.2126 + pixels[:, 1] * 0.7152 + pixels[:, 2] * 0.0722


def _hex_luma(color: str | None) -> float:
    if not color:
        return 0.0
    value = color.strip().lstrip("#")
    rgb = np.array([int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)], dtype=np.int16)
    return _luma(rgb)


def _fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")
