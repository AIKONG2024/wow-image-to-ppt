from __future__ import annotations

from PIL import Image, ImageDraw

from .models import BBox


def erase_regions(image: Image.Image, image_bbox: BBox, erase_boxes: list[BBox]) -> Image.Image:
    if not erase_boxes or image.width == 0 or image.height == 0:
        return image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for box in erase_boxes:
        region = _local_region(image.size, image_bbox, box)
        if region is not None:
            draw.rectangle(region, fill=255)
    if mask.getbbox() is None:
        return image
    if _mask_coverage(mask) > 0.28:
        return _fill_from_edge_color(image, mask)
    try:
        return _inpaint(image, mask)
    except Exception:
        return _fill_from_edge_color(image, mask)


def _inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    import cv2
    import numpy as np

    rgba = image.convert("RGBA")
    rgb = np.asarray(rgba.convert("RGB"))
    mask_array = np.asarray(mask)
    inpainted = cv2.inpaint(rgb, mask_array, 3, cv2.INPAINT_TELEA)
    alpha = np.asarray(rgba.getchannel("A"))
    result = np.dstack([inpainted, alpha])
    return Image.fromarray(result, mode="RGBA")


def _fill_from_edge_color(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    fill = _sample_light_background_color(rgba, mask)
    draw = ImageDraw.Draw(rgba)
    for region in _mask_regions(mask):
        draw.rectangle(region, fill=fill)
    return rgba


def _sample_light_background_color(image: Image.Image, mask: Image.Image) -> tuple[int, int, int, int]:
    pixels = image.load()
    mask_pixels = mask.load()
    step = max(1, min(image.width, image.height) // 96)
    sample = []
    for y in range(0, image.height, step):
        for x in range(0, image.width, step):
            if mask_pixels[x, y] == 0:
                color = pixels[x, y]
                luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                sample.append((luma, color))
    if not sample:
        return (255, 255, 255, 255)
    sample.sort(key=lambda item: item[0])
    return sample[int(len(sample) * 0.82)][1]


def _mask_coverage(mask: Image.Image) -> float:
    pixels = mask.histogram()
    covered = sum(count for value, count in enumerate(pixels) if value)
    return covered / max(1, mask.width * mask.height)


def _mask_regions(mask: Image.Image) -> list[tuple[int, int, int, int]]:
    bbox = mask.getbbox()
    return [bbox] if bbox else []


def _local_region(size: tuple[int, int], image_bbox: BBox, box: BBox) -> tuple[int, int, int, int] | None:
    width, height = size
    x1 = max(0, int(round(box.x - image_bbox.x)) - 2)
    y1 = max(0, int(round(box.y - image_bbox.y)) - 2)
    x2 = min(width, int(round(box.x + box.width - image_bbox.x)) + 2)
    y2 = min(height, int(round(box.y + box.height - image_bbox.y)) + 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
