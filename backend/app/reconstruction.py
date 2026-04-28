from __future__ import annotations

import uuid

import numpy as np
from PIL import Image

from .models import BBox, Component, Project, PptPrimitive


def reconstruct_project(project: Project, source: Image.Image) -> list[PptPrimitive]:
    visible_components = [component for component in project.components if not component.hidden]
    text_components = [component for component in visible_components if component.type == "text"]
    primitives: list[PptPrimitive] = []
    frame_components = [
        component
        for component in project.components
        if not component.hidden or _is_thin_rule(component, source.size)
    ]
    frame_primitives, consumed_component_ids = _frame_primitives(frame_components, source)
    synthetic_outlines = [
        component
        for component in visible_components
        if component.type == "shape"
        and component.source in {"synthetic-frame-shape", "synthetic-chart-panel-shape"}
    ]
    frame_primitives = [
        primitive
        for primitive in frame_primitives
        if not _has_equivalent_synthetic_outline(primitive.bbox, synthetic_outlines)
    ]
    primitives.extend(frame_primitives)

    for component in visible_components:
        if component.id in consumed_component_ids:
            continue
        if component.type == "text":
            if component.text:
                primitives.append(_textbox_primitive(component))
            continue
        if component.type in {"arrow", "line"}:
            primitives.append(_line_primitive(component, source))
            continue
        if _is_thin_rule(component, source.size):
            primitives.append(_line_primitive(component, source))
            continue
        if component.type == "shape":
            if component.source in {"synthetic-frame-shape", "synthetic-chart-panel-shape"}:
                primitives.append(_synthetic_outline_shape_primitive(component, source))
                continue
            if _is_fixed_header_background(component, source.size):
                primitives.append(_picture_primitive(component))
                continue
            stacked = _stacked_shape_picture_primitives(component, source, text_components, visible_components)
            if stacked is not None:
                primitives.extend(stacked)
                continue
            shape = _shape_primitive(component, source)
            primitives.append(shape if shape is not None else _picture_primitive(component))
            continue
        stacked = _stacked_shape_picture_primitives(component, source, text_components, visible_components)
        if stacked is not None:
            primitives.extend(stacked)
            continue
        primitives.append(_picture_primitive(component))

    return sorted(primitives, key=lambda primitive: (primitive.z_index, primitive.bbox.y, primitive.bbox.x))


def _textbox_primitive(component: Component) -> PptPrimitive:
    return PptPrimitive(
        id=_primitive_id("textbox"),
        kind="textbox",
        bbox=component.bbox,
        source_component_id=component.id,
        source_component_type=component.type,
        text=component.text,
        z_index=3000,
    )


def _shape_primitive(component: Component, source: Image.Image) -> PptPrimitive | None:
    crop = _component_crop(source, component.bbox)
    style = _shape_style(crop)
    if style is None:
        return None
    fill_color, line_color, line_width = style
    return PptPrimitive(
        id=_primitive_id("shape"),
        kind="shape",
        bbox=component.bbox,
        source_component_id=component.id,
        source_component_type=component.type,
        fill_color=fill_color,
        line_color=line_color,
        line_width=line_width,
        z_index=1000,
    )


def _synthetic_outline_shape_primitive(component: Component, source: Image.Image) -> PptPrimitive:
    crop = _component_crop(source, component.bbox)
    color = _outline_color_from_outer_band(crop) or _dominant_nonwhite_color(np.asarray(crop.convert("RGB"))) or "111827"
    return PptPrimitive(
        id=_primitive_id("shape"),
        kind="shape",
        bbox=component.bbox,
        source_component_id=component.id,
        source_component_type=component.type,
        fill_color=None,
        line_color=color,
        line_width=1.25,
        z_index=900,
    )


def _line_primitive(component: Component, source: Image.Image) -> PptPrimitive:
    color = _dominant_nonwhite_color(np.asarray(_component_crop(source, component.bbox).convert("RGB"))) or "111827"
    x1, y1, x2, y2 = _line_endpoints(component.bbox)
    return PptPrimitive(
        id=_primitive_id(component.type),
        kind="arrow" if component.type == "arrow" else "line",
        bbox=component.bbox,
        source_component_id=component.id,
        source_component_type=component.type,
        line_color=color,
        line_width=max(1.0, min(4.0, min(component.bbox.width, component.bbox.height) * 0.55)),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        z_index=2200,
    )


def _picture_primitive(component: Component) -> PptPrimitive:
    return PptPrimitive(
        id=_primitive_id("picture"),
        kind="picture",
        bbox=component.bbox,
        source_component_id=component.id,
        source_component_type=component.type,
        asset_path=component.asset_path,
        mask_path=component.mask_path,
        z_index=1800,
    )


def _stacked_shape_picture_primitives(
    component: Component,
    source: Image.Image,
    text_components: list[Component],
    visible_components: list[Component],
) -> list[PptPrimitive] | None:
    if component.type == "shape" and _has_contained_visual_child(component, visible_components):
        return None
    split = layered_shape_picture_split(component, source, text_components)
    if split is None:
        return None
    fill_color, line_color, line_width, inner_bbox = split

    return [
        PptPrimitive(
            id=_primitive_id("shape"),
            kind="shape",
            bbox=component.bbox,
            source_component_id=component.id,
            source_component_type=component.type,
            fill_color=fill_color,
            line_color=line_color,
            line_width=line_width,
            z_index=1000,
        ),
        PptPrimitive(
            id=_primitive_id("picture"),
            kind="picture",
            bbox=inner_bbox,
            source_component_id=component.id,
            source_component_type=component.type,
            z_index=1800,
        ),
    ]


def _has_contained_visual_child(component: Component, components: list[Component]) -> bool:
    for candidate in components:
        if candidate.id == component.id or candidate.hidden:
            continue
        if candidate.type not in {"image", "icon", "diagram", "chart"}:
            continue
        if candidate.bbox.width * candidate.bbox.height >= component.bbox.width * component.bbox.height * 0.72:
            continue
        if _bbox_containment_ratio(candidate.bbox, component.bbox) >= 0.75:
            return True
    return False


def layered_shape_picture_split(
    component: Component,
    source: Image.Image,
    text_components: list[Component],
) -> tuple[str, str | None, float | None, BBox] | None:
    if component.type not in {"image", "icon", "diagram", "shape", "unknown"}:
        return None

    crop = _component_crop(source, component.bbox)
    rgba = np.asarray(crop.convert("RGBA"))
    if rgba.size == 0:
        return None

    valid_mask = rgba[:, :, 3] > 0
    style = _layered_background_style(rgba, valid_mask)
    if style is None:
        return None
    fill_color, line_color, line_width = style

    foreground_mask = _foreground_different_from_fill(rgba, valid_mask, fill_color)
    _remove_text_regions(foreground_mask, component.bbox, text_components)
    inner_bbox = _foreground_bbox(component.bbox, foreground_mask)
    if inner_bbox is None:
        return None
    if _covers_most_of_component(inner_bbox, component.bbox):
        return None
    return fill_color, line_color, line_width, inner_bbox


def _frame_primitives(components: list[Component], source: Image.Image) -> tuple[list[PptPrimitive], set[str]]:
    groups = _thin_vertical_groups(components, source.size)
    horizontal_groups = _thin_horizontal_groups(components, source.size)
    consumed: set[str] = set()
    primitives: list[PptPrimitive] = []
    for left in groups:
        if consumed.intersection(left["ids"]):
            continue
        candidates = [
            right
            for right in groups
            if right["bbox"].x > left["bbox"].x + 24
            and not consumed.intersection(right["ids"])
            and _vertical_overlap_ratio(left["bbox"], right["bbox"]) >= 0.65
        ]
        if not candidates:
            continue
        right = min(candidates, key=lambda item: _frame_pair_score(left["bbox"], item["bbox"], source.width))
        bbox = _bbox_union([left["bbox"], right["bbox"]])
        horizontal_edges = _matching_horizontal_frame_edges(bbox, horizontal_groups, consumed)
        bbox = _bbox_union([bbox, *[edge["bbox"] for edge in horizontal_edges]])
        if bbox.width < source.width * 0.08 or bbox.height < source.height * 0.08:
            continue
        color = _dominant_nonwhite_color(np.asarray(_component_crop(source, left["bbox"]).convert("RGB"))) or "111827"
        source_ids = [*left["ids"], *right["ids"]]
        for edge in horizontal_edges:
            source_ids.extend(edge["ids"])
        primitives.append(
            PptPrimitive(
                id=_primitive_id("frame"),
                kind="shape",
                bbox=bbox,
                source_component_id="+".join(source_ids),
                source_component_type="shape",
                fill_color=None,
                line_color=color,
                line_width=max(1.0, min(2.5, left["bbox"].width)),
                z_index=900,
            )
        )
        consumed.update(left["ids"])
        consumed.update(right["ids"])
        for edge in horizontal_edges:
            consumed.update(edge["ids"])
    return primitives, consumed


def _thin_vertical_groups(components: list[Component], image_size: tuple[int, int]) -> list[dict[str, object]]:
    verticals = [component for component in components if _is_thin_vertical(component, image_size)]
    verticals.sort(key=lambda component: (round(component.bbox.x), component.bbox.y))
    groups: list[dict[str, object]] = []
    for component in verticals:
        appended = False
        for group in groups:
            bbox = group["bbox"]
            if abs(component.bbox.x - bbox.x) <= 6 and _vertical_gap(component.bbox, bbox) <= 32:
                group["bbox"] = _bbox_union([bbox, component.bbox])
                group["ids"].append(component.id)
                appended = True
                break
        if not appended:
            groups.append({"bbox": component.bbox, "ids": [component.id]})
    return groups


def _thin_horizontal_groups(components: list[Component], image_size: tuple[int, int]) -> list[dict[str, object]]:
    horizontals = [component for component in components if _is_thin_horizontal(component, image_size)]
    horizontals.sort(key=lambda component: (round(component.bbox.y), component.bbox.x))
    groups: list[dict[str, object]] = []
    for component in horizontals:
        appended = False
        for group in groups:
            bbox = group["bbox"]
            if abs(component.bbox.y - bbox.y) <= 6 and _horizontal_gap(component.bbox, bbox) <= 32:
                group["bbox"] = _bbox_union([bbox, component.bbox])
                group["ids"].append(component.id)
                appended = True
                break
        if not appended:
            groups.append({"bbox": component.bbox, "ids": [component.id]})
    return groups


def _matching_horizontal_frame_edges(
    bbox: BBox,
    horizontal_groups: list[dict[str, object]],
    consumed: set[str],
) -> list[dict[str, object]]:
    top_candidates: list[dict[str, object]] = []
    bottom_candidates: list[dict[str, object]] = []
    center_y = bbox.y + bbox.height / 2
    for group in horizontal_groups:
        if consumed.intersection(group["ids"]):
            continue
        edge = group["bbox"]
        if _horizontal_overlap_ratio(edge, bbox) < 0.65:
            continue
        edge_y = edge.y + edge.height / 2
        if abs(edge_y - bbox.y) <= 10:
            top_candidates.append(group)
        elif edge_y <= center_y and abs(edge_y - bbox.y) <= max(14, bbox.height * 0.12):
            top_candidates.append(group)
        if abs(edge_y - (bbox.y + bbox.height)) <= 10:
            bottom_candidates.append(group)
        elif edge_y >= center_y and abs(edge_y - (bbox.y + bbox.height)) <= max(14, bbox.height * 0.12):
            bottom_candidates.append(group)

    edges: list[dict[str, object]] = []
    if top_candidates:
        edges.append(min(top_candidates, key=lambda item: abs((item["bbox"].y + item["bbox"].height / 2) - bbox.y)))
    if bottom_candidates:
        bottom = min(
            bottom_candidates,
            key=lambda item: abs((item["bbox"].y + item["bbox"].height / 2) - (bbox.y + bbox.height)),
        )
        if bottom not in edges:
            edges.append(bottom)
    return edges


def _is_fixed_header_background(component: Component, image_size: tuple[int, int]) -> bool:
    image_width, image_height = image_size
    return (
        component.bbox.x <= 2
        and component.bbox.y <= 2
        and component.bbox.width >= image_width * 0.85
        and component.bbox.height <= image_height * 0.22
    )


def _is_thin_rule(component: Component, image_size: tuple[int, int]) -> bool:
    return _is_thin_vertical(component, image_size) or _is_thin_horizontal(component, image_size)


def _is_thin_vertical(component: Component, image_size: tuple[int, int]) -> bool:
    image_width, image_height = image_size
    return (
        component.type in {"icon", "shape", "unknown"}
        and component.source.startswith("opencv")
        and component.bbox.width <= max(5, image_width * 0.004)
        and component.bbox.height >= image_height * 0.08
    )


def _is_thin_horizontal(component: Component, image_size: tuple[int, int]) -> bool:
    image_width, image_height = image_size
    return (
        component.type in {"icon", "shape", "unknown"}
        and component.source.startswith("opencv")
        and component.bbox.height <= max(5, image_height * 0.006)
        and component.bbox.width >= image_width * 0.08
    )


def _shape_style(crop: Image.Image) -> tuple[str | None, str | None, float | None] | None:
    rgba = np.asarray(crop.convert("RGBA"))
    if rgba.size == 0:
        return None

    valid_mask = rgba[:, :, 3] > 0
    outer_fill = _solid_fill_from_outer_band(rgba, valid_mask)
    if outer_fill is not None:
        return outer_fill, None, None
    card_style = _outer_background_style(rgba, valid_mask)
    if card_style is not None and card_style[1] is not None:
        return card_style

    style = _style_from_valid_pixels(rgba, valid_mask)
    if style is not None:
        return style
    return None


def _layered_background_style(rgba: np.ndarray, valid_mask: np.ndarray) -> tuple[str, str | None, float | None] | None:
    solid_fill = _solid_fill_from_outer_band(rgba, valid_mask)
    if solid_fill is not None:
        return solid_fill, None, None
    return _outer_background_style(rgba, valid_mask)


def _outer_background_style(rgba: np.ndarray, valid_mask: np.ndarray) -> tuple[str, str | None, float | None] | None:
    height, width = valid_mask.shape
    if height < 8 or width < 8:
        return None
    band = max(3, min(14, int(round(min(width, height) * 0.16))))
    outer_mask = np.zeros_like(valid_mask, dtype=bool)
    outer_mask[:band, :] = True
    outer_mask[-band:, :] = True
    outer_mask[:, :band] = True
    outer_mask[:, -band:] = True
    outer_mask &= valid_mask
    if int(np.count_nonzero(outer_mask)) < 32:
        return None

    outer_pixels = rgba[:, :, :3][outer_mask]
    background = _dominant_color(outer_pixels)
    if _color_dominance_ratio(outer_pixels, background, tolerance=14) < 0.45:
        return None

    background_rgb = np.array(_hex_to_rgb(background), dtype=np.int16)
    distance = np.max(np.abs(outer_pixels.astype(np.int16) - background_rgb), axis=1)
    line_pixels = outer_pixels[distance > 28]
    line_color = None
    line_width = None
    if len(line_pixels) / max(1, len(outer_pixels)) >= 0.025:
        line_color = _saturated_outline_color(line_pixels) or _dominant_color(line_pixels)
        line_width = 1.25
    return background, line_color, line_width


def _solid_fill_from_outer_band(rgba: np.ndarray, valid_mask: np.ndarray) -> str | None:
    height, width = valid_mask.shape
    if height < 4 or width < 4:
        return None
    band = max(2, min(10, int(round(min(width, height) * 0.18))))
    outer_mask = np.zeros_like(valid_mask, dtype=bool)
    outer_mask[:band, :] = True
    outer_mask[-band:, :] = True
    outer_mask[:, :band] = True
    outer_mask[:, -band:] = True
    outer_mask &= valid_mask
    if int(np.count_nonzero(outer_mask)) < 20:
        return None

    full_pixels = rgba[:, :, :3][valid_mask]
    full_nonwhite_ratio = float(np.count_nonzero(_nonwhite_pixels(full_pixels))) / max(1, len(full_pixels))
    if full_nonwhite_ratio < 0.25:
        return None

    outer_pixels = rgba[:, :, :3][outer_mask]
    outer_nonwhite_mask = _nonwhite_pixels(outer_pixels)
    outer_nonwhite_ratio = float(np.count_nonzero(outer_nonwhite_mask)) / max(1, len(outer_pixels))
    if outer_nonwhite_ratio < 0.55:
        return None

    outer_nonwhite_pixels = outer_pixels[outer_nonwhite_mask]
    dominant = _dominant_color(outer_nonwhite_pixels)
    if _dominance_ratio(outer_nonwhite_pixels, dominant) < 0.42:
        return None
    return dominant


def _style_from_valid_pixels(rgba: np.ndarray, valid_mask: np.ndarray) -> tuple[str | None, str | None, float | None] | None:
    if int(np.count_nonzero(valid_mask)) < 20:
        return None

    rgb = rgba[:, :, :3]
    valid_pixels = rgb[valid_mask]
    nonwhite_mask = _nonwhite_pixels(valid_pixels)
    nonwhite_ratio = float(np.count_nonzero(nonwhite_mask)) / max(1, len(valid_pixels))
    if nonwhite_ratio < 0.01:
        return None

    nonwhite_pixels = valid_pixels[nonwhite_mask]
    dominant = _dominant_color(nonwhite_pixels)
    dominance = _dominance_ratio(nonwhite_pixels, dominant)
    if nonwhite_ratio >= 0.35 and dominance >= 0.45:
        return dominant, None, None

    global_nonwhite = np.zeros(valid_mask.shape, dtype=bool)
    global_nonwhite[valid_mask] = nonwhite_mask
    if _is_outline_like(global_nonwhite):
        return None, dominant, 1.25

    return None


def _foreground_different_from_fill(rgba: np.ndarray, valid_mask: np.ndarray, fill_color: str) -> np.ndarray:
    fill_rgb = np.array(_hex_to_rgb(fill_color), dtype=np.int16)
    rgb = rgba[:, :, :3].astype(np.int16)
    distance = np.max(np.abs(rgb - fill_rgb), axis=2)
    foreground = (distance > 28) & valid_mask
    height, width = foreground.shape
    if height >= 6 and width >= 6:
        pad = max(1, min(3, round(min(width, height) * 0.04)))
        foreground[:pad, :] = False
        foreground[-pad:, :] = False
        foreground[:, :pad] = False
        foreground[:, -pad:] = False
    return foreground


def _remove_text_regions(mask: np.ndarray, component_bbox: BBox, text_components: list[Component]) -> None:
    for text in text_components:
        if not _intersects(component_bbox, text.bbox):
            continue
        x1 = max(0, int(np.floor(text.bbox.x - component_bbox.x)) - 2)
        y1 = max(0, int(np.floor(text.bbox.y - component_bbox.y)) - 2)
        x2 = min(mask.shape[1], int(np.ceil(text.bbox.x + text.bbox.width - component_bbox.x)) + 2)
        y2 = min(mask.shape[0], int(np.ceil(text.bbox.y + text.bbox.height - component_bbox.y)) + 2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = False


def _foreground_bbox(component_bbox: BBox, foreground_mask: np.ndarray) -> BBox | None:
    if int(np.count_nonzero(foreground_mask)) < 20:
        return None
    rows, columns = np.where(foreground_mask)
    if len(rows) == 0 or len(columns) == 0:
        return None
    x1 = int(columns.min())
    x2 = int(columns.max()) + 1
    y1 = int(rows.min())
    y2 = int(rows.max()) + 1
    width = x2 - x1
    height = y2 - y1
    if width < 4 or height < 4:
        return None
    return BBox(
        x=component_bbox.x + x1,
        y=component_bbox.y + y1,
        width=float(width),
        height=float(height),
    )


def _covers_most_of_component(inner: BBox, outer: BBox) -> bool:
    width_ratio = inner.width / max(1.0, outer.width)
    height_ratio = inner.height / max(1.0, outer.height)
    area_ratio = (inner.width * inner.height) / max(1.0, outer.width * outer.height)
    return (width_ratio >= 0.82 and height_ratio >= 0.82) or area_ratio >= 0.70


def _component_crop(source: Image.Image, bbox: BBox) -> Image.Image:
    return source.crop(
        (
            max(0, round(bbox.x)),
            max(0, round(bbox.y)),
            min(source.width, round(bbox.x + bbox.width)),
            min(source.height, round(bbox.y + bbox.height)),
        )
    )


def _dominant_nonwhite_color(rgb: np.ndarray) -> str | None:
    if rgb.size == 0:
        return None
    pixels = rgb.reshape(-1, 3)
    nonwhite = pixels[_nonwhite_pixels(pixels)]
    if len(nonwhite) == 0:
        return None
    return _dominant_color(nonwhite)


def _outline_color_from_outer_band(crop: Image.Image) -> str | None:
    rgba = np.asarray(crop.convert("RGBA"))
    if rgba.size == 0:
        return None
    valid_mask = rgba[:, :, 3] > 0
    height, width = valid_mask.shape
    if height < 4 or width < 4:
        return None
    band = max(2, min(10, int(round(min(width, height) * 0.08))))
    outer_mask = np.zeros_like(valid_mask, dtype=bool)
    outer_mask[:band, :] = True
    outer_mask[-band:, :] = True
    outer_mask[:, :band] = True
    outer_mask[:, -band:] = True
    outer_mask &= valid_mask
    pixels = rgba[:, :, :3][outer_mask]
    if len(pixels) == 0:
        return None
    nonwhite = pixels[_nonwhite_pixels(pixels)]
    if len(nonwhite) < 4:
        return None
    return _dominant_color(nonwhite)


def _dominant_color(pixels: np.ndarray) -> str:
    if len(pixels) > 30000:
        stride = max(1, len(pixels) // 30000)
        pixels = pixels[::stride]
    quantized = (pixels // 8) * 8
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    winner = colors[int(np.argmax(counts))]
    bucket = np.all(quantized == winner, axis=1)
    representative = np.median(pixels[bucket] if np.any(bucket) else pixels, axis=0)
    return _rgb_to_hex(tuple(int(round(value)) for value in representative))


def _saturated_outline_color(pixels: np.ndarray) -> str | None:
    if len(pixels) < 8:
        return None
    values = pixels.astype(np.int16)
    saturation = values.max(axis=1) - values.min(axis=1)
    luma = values[:, 0] * 0.2126 + values[:, 1] * 0.7152 + values[:, 2] * 0.0722
    candidates = pixels[(saturation >= 45) & (luma < 245)]
    if len(candidates) < max(8, int(len(pixels) * 0.08)):
        return None
    color = _dominant_color(candidates)
    rgb = _hex_to_rgb(color)
    if max(rgb) - min(rgb) < 55:
        return None
    return color


def _dominance_ratio(pixels: np.ndarray, color: str) -> float:
    return _color_dominance_ratio(pixels, color, tolerance=8)


def _color_dominance_ratio(pixels: np.ndarray, color: str, tolerance: int) -> float:
    target = np.array(_hex_to_rgb(color), dtype=np.int16)
    distances = np.max(np.abs(pixels.astype(np.int16) - target), axis=1)
    return float(np.count_nonzero(distances <= tolerance)) / max(1, len(pixels))


def _nonwhite_pixels(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.int16)
    distance_from_white = np.max(np.abs(values - 255), axis=1)
    channel_spread = values.max(axis=1) - values.min(axis=1)
    return (distance_from_white > 16) | (channel_spread > 12)


def _is_outline_like(nonwhite_mask: np.ndarray) -> bool:
    nonwhite_area = int(np.count_nonzero(nonwhite_mask))
    if nonwhite_area == 0:
        return False
    height, width = nonwhite_mask.shape
    border = max(2, int(round(min(width, height) * 0.16)))
    border_mask = np.zeros_like(nonwhite_mask, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    return float(np.count_nonzero(nonwhite_mask & border_mask)) / nonwhite_area >= 0.6


def _line_endpoints(bbox: BBox) -> tuple[float, float, float, float]:
    if bbox.width >= bbox.height:
        y = bbox.y + bbox.height / 2
        return bbox.x, y, bbox.x + bbox.width, y
    x = bbox.x + bbox.width / 2
    return x, bbox.y, x, bbox.y + bbox.height


def _bbox_union(boxes: list[BBox]) -> BBox:
    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.x + box.width for box in boxes)
    y2 = max(box.y + box.height for box in boxes)
    return BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def _vertical_gap(left: BBox, right: BBox) -> float:
    if left.y > right.y:
        left, right = right, left
    return max(0.0, right.y - (left.y + left.height))


def _horizontal_gap(left: BBox, right: BBox) -> float:
    if left.x > right.x:
        left, right = right, left
    return max(0.0, right.x - (left.x + left.width))


def _vertical_overlap_ratio(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    shorter = max(1.0, min(left.height, right.height))
    return overlap / shorter


def _horizontal_overlap_ratio(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    wider_target = max(1.0, right.width)
    return overlap / wider_target


def _has_equivalent_synthetic_outline(bbox: BBox, synthetic_outlines: list[Component]) -> bool:
    for component in synthetic_outlines:
        if _bbox_area_similarity(component.bbox, bbox) < 0.78:
            continue
        if _overlap_ratio(component.bbox, bbox) >= 0.86:
            return True
    return False


def _bbox_area_similarity(left: BBox, right: BBox) -> float:
    left_area = max(1.0, left.width * left.height)
    right_area = max(1.0, right.width * right.height)
    return min(left_area, right_area) / max(left_area, right_area)


def _overlap_ratio(left: BBox, right: BBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = max(1.0, min(left.width * left.height, right.width * right.height))
    return intersection / smaller


def _frame_pair_score(left: BBox, right: BBox, image_width: int) -> tuple[float, float, float]:
    height_similarity = min(left.height, right.height) / max(left.height, right.height, 1.0)
    center_delta = abs((left.y + left.height / 2) - (right.y + right.height / 2)) / max(left.height, right.height, 1.0)
    distance_penalty = (right.x - left.x) / max(image_width * 8, 1)
    return (1.0 - height_similarity, center_delta, distance_penalty)


def _intersects(left: BBox, right: BBox) -> bool:
    return not (
        left.x + left.width <= right.x
        or right.x + right.width <= left.x
        or left.y + left.height <= right.y
        or right.y + right.height <= left.y
    )


def _bbox_containment_ratio(inner: BBox, outer: BBox) -> float:
    x1 = max(inner.x, outer.x)
    y1 = max(inner.y, outer.y)
    x2 = min(inner.x + inner.width, outer.x + outer.width)
    y2 = min(inner.y + inner.height, outer.y + outer.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inner_area = max(1.0, inner.width * inner.height)
    return intersection / inner_area


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    red, green, blue = color
    return f"{red:02X}{green:02X}{blue:02X}"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.strip().lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _primitive_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"
