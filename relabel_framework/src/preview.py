from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_label_array(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def colorize_label(label_arr: np.ndarray, palette: list[list[int]]) -> np.ndarray:
    palette_arr = np.array(palette, dtype=np.uint8)
    if palette_arr.size == 0:
        raise ValueError("Palette cannot be empty")
    safe = np.clip(label_arr.astype(np.int64), 0, len(palette_arr) - 1)
    return palette_arr[safe]


def remap_label_array(label_arr: np.ndarray, groups: list[int], unknown_value: int = 0) -> np.ndarray:
    result = np.full(label_arr.shape, unknown_value, dtype=np.uint8)
    for old_value, new_value in enumerate(groups):
        result[label_arr == old_value] = int(new_value)
    return result


def overlay_mask(raw_image: Image.Image, label_arr: np.ndarray, palette: list[list[int]], alpha: float = 0.58) -> Image.Image:
    raw = raw_image.convert("RGB")
    mask = Image.fromarray(colorize_label(label_arr, palette)).resize(raw.size, Image.NEAREST)
    alpha = min(max(float(alpha), 0.0), 1.0)
    return Image.blend(raw, mask, alpha)


def _text_color_for_fill(color: list[int]) -> tuple[int, int, int]:
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return (0, 0, 0) if luminance > 145 else (255, 255, 255)


def draw_class_labels(
    image: Image.Image,
    label_arr: np.ndarray,
    class_names: list[str],
    palette: list[list[int]],
    min_pixels: int = 700,
) -> Image.Image:
    labeled = image.convert("RGB").copy()
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    scale_x = labeled.size[0] / label_arr.shape[1]
    scale_y = labeled.size[1] / label_arr.shape[0]

    for class_id in sorted(int(value) for value in np.unique(label_arr)):
        if class_id < 0 or class_id >= len(class_names) or class_id >= len(palette):
            continue
        ys, xs = np.where(label_arr == class_id)
        if len(xs) < min_pixels:
            continue
        x = int(float(np.median(xs)) * scale_x)
        y = int(float(np.median(ys)) * scale_y)
        text = class_names[class_id]
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 4
        rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        draw.rounded_rectangle(rect, radius=4, fill=tuple(palette[class_id]), outline=(0, 0, 0), width=1)
        draw.text((x, y), text, font=font, fill=_text_color_for_fill(palette[class_id]))
    return labeled


def make_relabel_preview(
    image_path: str | Path,
    label_path: str | Path,
    groups: list[int],
    source_palette: list[list[int]],
    target_palette: list[list[int]],
    target_names: list[str],
    alpha: float = 0.58,
) -> dict[str, Image.Image]:
    raw = Image.open(image_path).convert("RGB")
    source_labels = load_label_array(label_path)
    relabeled = remap_label_array(source_labels, groups)

    source_mask = Image.fromarray(colorize_label(source_labels, source_palette)).resize(raw.size, Image.NEAREST)
    relabeled_mask = Image.fromarray(colorize_label(relabeled, target_palette)).resize(raw.size, Image.NEAREST)
    overlay = overlay_mask(raw, relabeled, target_palette, alpha=alpha)
    overlay = draw_class_labels(overlay, relabeled, target_names, target_palette)

    return {
        "raw": raw,
        "source_mask": source_mask,
        "relabeled_mask": relabeled_mask,
        "overlay": overlay,
    }


def resize_label_for_preview(label_arr: np.ndarray, max_width: int = 1200) -> np.ndarray:
    if label_arr.shape[1] <= max_width:
        return label_arr
    scale = max_width / label_arr.shape[1]
    new_size = (max_width, max(1, int(label_arr.shape[0] * scale)))
    img = Image.fromarray(label_arr.astype(np.uint8))
    return np.array(img.resize(new_size, Image.NEAREST))


def resize_image_for_preview(image: Image.Image, max_width: int = 1200) -> Image.Image:
    if image.width <= max_width:
        return image
    scale = max_width / image.width
    return image.resize((max_width, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
